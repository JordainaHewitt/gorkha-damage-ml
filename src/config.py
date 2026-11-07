"""Feature definitions for the Ghimire et al. (2022) reproduction.

Column names follow `csv_building_structure.csv` from the 2015 Nepal
Earthquake Open Data Portal (http://eq2015.npc.gov.np/). Verify them against
your download; if a name differs, change it here only.
"""

# Paper Table 1, numeric building parameters. Use the PRE-earthquake versions:
# the *_post_eq columns were measured after the damage and would leak the label.
NUMERIC = [
    "count_floors_pre_eq",
    "age_building",
    "plinth_area_sq_ft",
    "height_ft_pre_eq",
]

# Paper Table 1, categorical parameters -> one-hot encoded.
CATEGORICAL = [
    "land_surface_condition",  # ground slope: flat / moderate / steep
    "roof_type",               # light timber, heavy timber, RC
    "position",                # not attached, attached 1/2/3 sides
]

# Superstructure material is already multi-hot in the raw data: one 0/1 column
# per material, and a building can use several (percentages in Table 1 sum
# to >100% for this reason). No encoding needed.
MATERIALS = [
    "has_superstructure_adobe_mud",
    "has_superstructure_mud_mortar_stone",
    "has_superstructure_stone_flag",
    "has_superstructure_cement_mortar_stone",
    "has_superstructure_mud_mortar_brick",
    "has_superstructure_cement_mortar_brick",
    "has_superstructure_timber",
    "has_superstructure_bamboo",
    "has_superstructure_rc_non_engineered",
    "has_superstructure_rc_engineered",
    "has_superstructure_other",
]

# The paper's "basic-features setting": 4 parameters available from a census.
BASIC = NUMERIC

TARGET = "damage_grade"
LOCATION_KEYS = ["district_id", "vdcmun_id", "ward_id"]

# Paper: age ranges 1-200 yr; 5731 buildings had missing age.
AGE_MAX_VALID = 200

# Traffic-light grouping (paper Sec. "Damage prediction using test dataset"):
# S = DG1+DG2, M = DG3, H = DG4+DG5.
TL_MAP = {1: 1, 2: 1, 3: 2, 4: 3, 5: 3}

SEED = 42
