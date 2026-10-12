"""spaCy EntityRuler patterns for the 8 Task 2 entity types (rule-only baseline and pre-annotation)."""
import spacy

LABELS = ["RUNWAY", "FLIGHT_LEVEL", "ALTITUDE", "HEADING", "FREQUENCY", "CALLSIGN", "AIRCRAFT_TYPE", "AIRPORT"]
RWY_NUM = r"^(0?[1-9]|[12]\d|3[0-6])[lrc]?$"
ACFT = (r"^(b7[0-8]7|a3[0-8]\d|a2[12]\d|crj\d{0,4}|erj\d{0,3}|emb\d{3}|e1[79]\d|c-?1\d{2}|c-?[2-5]\d{2}|pa-?\d{2}|md-?\d{2}|"
        r"dc-?\d{1,2}|be-?\d{2,4}|dhc-?\d|da-?\d{2}|atr-?\d{2}|sr-?2[02]|pc-?12|cl-?\d{2,3})(-\d{1,3}(er|lr)?)?$")
PATTERNS = [
    {"label": "RUNWAY", "pattern": [{"LOWER": {"IN": ["rwy", "rwys", "runway", "runways"]}}, {"LOWER": {"REGEX": RWY_NUM}}]},
    {"label": "RUNWAY", "pattern": [{"LOWER": {"REGEX": r"^rwy(0?[1-9]|[12]\d|3[0-6])[lrc]?$"}}]},
    {"label": "FLIGHT_LEVEL", "pattern": [{"LOWER": {"REGEX": r"^fl\d{2,3}$"}}]},
    {"label": "FLIGHT_LEVEL", "pattern": [{"LOWER": "fl"}, {"LOWER": {"REGEX": r"^\d{2,3}$"}}]},
    {"label": "FLIGHT_LEVEL", "pattern": [{"LOWER": "flight"}, {"LOWER": "level"}, {"LOWER": {"REGEX": r"^\d{2,3}$"}}]},
    {"label": "ALTITUDE", "pattern": [{"LOWER": {"REGEX": r"^\d{1,2}[,;]\d{3}$|^\d{3,5}$"}}, {"LOWER": {"IN": ["ft", "feet", "foot", "msl", "agl"]}}]},
    {"label": "ALTITUDE", "pattern": [{"LOWER": {"REGEX": r"^\d{1,2}$"}}, {"ORTH": {"IN": [",", ";"]}}, {"LOWER": {"REGEX": r"^\d{3}$"}},
                                      {"LOWER": {"IN": ["ft", "feet", "foot", "msl", "agl"]}}]},
    {"label": "ALTITUDE", "pattern": [{"LOWER": {"REGEX": r"^\d{3,5}(ft|')$"}}]},
    {"label": "HEADING", "pattern": [{"LOWER": {"IN": ["hdg", "heading"]}}, {"LOWER": "of", "OP": "?"}, {"LOWER": {"REGEX": r"^[0-3]\d{2}$"}}]},
    {"label": "FREQUENCY", "pattern": [{"TEXT": {"REGEX": r"^1[1-3]\d\.\d{1,3}$"}}]},
    {"label": "CALLSIGN", "pattern": [{"LOWER": {"IN": ["acr", "cpr", "sma", "mlt", "ltt", "acft", "plt", "spi"]}},
                                      {"LOWER": {"REGEX": r"^[xyz]\d{0,4}$"}}]},
    {"label": "AIRCRAFT_TYPE", "pattern": [{"LOWER": {"REGEX": ACFT}}]},
    {"label": "AIRPORT", "pattern": [{"LOWER": {"REGEX": r"^zzz\d?$"}}]},
    {"label": "AIRPORT", "pattern": [{"LOWER": {"REGEX": r"^k[a-z]{3}$"}}, {"LOWER": "airport"}]},
    {"label": "AIRPORT", "pattern": [{"LOWER": {"REGEX": r"^[a-z]{3}$", "NOT_IN": ["the", "and", "his", "her", "our", "new", "old", "off", "any", "one", "two"]}},
                                     {"LOWER": {"IN": ["airport", "arpt"]}}]},
    {"label": "AIRPORT", "pattern": [{"LOWER": {"REGEX": r"^x\d{2}$"}}]},
]


def build_nlp():
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns(PATTERNS)
    return nlp
