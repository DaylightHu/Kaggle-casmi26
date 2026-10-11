"""Scorer identity and explicit adduct chemistry; no uncharging/salt stripping."""
import re
from functools import lru_cache

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

METRIC_RDKIT = "2026.03.3"
ELECTRON = 0.000548579909065
ATOMS = {"C": 12.0, "H": 1.00782503223, "N": 14.00307400443,
         "O": 15.99491461957, "Na": 22.9897692820, "K": 38.9637064864,
         "Cl": 34.968852682, "S": 31.9720711744, "P": 30.97376199842,
         "Br": 78.9183376}
TEST_ADDUCTS = ["[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+",
                "[M+Na]+", "[M+K]+", "[M-H]-", "[M-H2O-H]-",
                "[M+CH2O2-H]-", "[M+Cl]-"]
ALIASES = {"ACN": "C2H3N", "FA": "CH2O2", "Hac": "C2H4O2",
           "HCOO": "CHO2", "CH3COO": "C2H3O2"}
TAUTOMERS = rdMolStandardize.TautomerEnumerator()


def check_rdkit(allow_mismatch=False):
    if rdBase.rdkitVersion != METRIC_RDKIT:
        message = f"RDKit {rdBase.rdkitVersion}; official identity requires {METRIC_RDKIT}."
        if not allow_mismatch:
            raise RuntimeError(message + " Install rdkit==2026.3.3. Use --allow-rdkit-mismatch only for development.")
        print("WARNING:", message, "This run is not an official-metric reproduction.")
    return rdBase.rdkitVersion


def formula_mass(formula):
    formula = ALIASES.get(formula, formula)
    pieces = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    if not pieces or "".join(e + n for e, n in pieces) != formula:
        raise ValueError(f"Unsupported adduct formula: {formula}")
    return sum(ATOMS[e] * int(n or "1") for e, n in pieces)


@lru_cache(maxsize=256)
def parse_adduct(adduct):
    """Return molecular multiplicity, absolute charge and ionic mass shift."""
    match = re.fullmatch(r"\[(\d*)M([^\]]*)\](\d*)([+-])", str(adduct).replace(" ", ""))
    if not match:
        raise ValueError(f"Unsupported adduct: {adduct}")
    multi, terms, charge, sign = match.groups()
    z = int(charge or "1")
    shift = -ELECTRON * z * (1 if sign == "+" else -1)
    pieces = re.findall(r"([+-])(\d*)([A-Za-z][A-Za-z0-9]*)", terms)
    if "".join(a+b+c for a,b,c in pieces) != terms:
        raise ValueError(f"Unsupported adduct terms: {adduct}")
    for operation, count, formula in pieces:
        shift += (1 if operation == "+" else -1) * int(count or "1") * formula_mass(formula)
    return int(multi or "1"), z, shift


def neutral_mass(precursor_mz, adduct):
    multi, charge, shift = parse_adduct(str(adduct))
    value = (float(precursor_mz) * charge - shift) / multi
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid neutral mass from {precursor_mz}, {adduct}")
    return value


def ion_mz(mass, adduct):
    multi, charge, shift = parse_adduct(adduct)
    return (mass * multi + shift) / charge


@lru_cache(maxsize=20000)
def scorer_key(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid SMILES: {str(smiles)[:100]}")
    mol = TAUTOMERS.Canonicalize(mol)
    key = Chem.MolToInchiKey(mol)
    if not key:
        raise ValueError("RDKit could not generate an InChIKey")
    return key.split("-")[0]


@lru_cache(maxsize=8)
def fingerprint_generator(bits, radius):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits, includeChirality=False)


def describe_structure(smiles, bits=2048, radius=2):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("Invalid or empty SMILES")
    mol = TAUTOMERS.Canonicalize(mol)
    key = Chem.MolToInchiKey(mol).split("-")[0]
    if len(key) != 14:
        raise ValueError("Invalid InChIKey14")
    Chem.RemoveStereochemistry(mol)
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    # isomericSmiles keeps isotopes after stereochemistry has been explicitly removed.
    values = np.zeros(bits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fingerprint_generator(bits, radius).GetFingerprint(mol), values)
    return {"key": key, "smiles": canonical,
            "mass": float(rdMolDescriptors.CalcExactMolWt(mol)),
            "formula": rdMolDescriptors.CalcMolFormula(mol)}, np.packbits(values, bitorder="little")
