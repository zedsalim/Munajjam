"""
Surah metadata model.
"""

from typing import Optional

from pydantic import BaseModel, Field

# Surah names in Arabic
SURAH_NAMES: dict[int, str] = {
    1: "الفاتحة",
    2: "البقرة",
    3: "آل عمران",
    4: "النساء",
    5: "المائدة",
    6: "الأنعام",
    7: "الأعراف",
    8: "الأنفال",
    9: "التوبة",
    10: "يونس",
    11: "هود",
    12: "يوسف",
    13: "الرعد",
    14: "إبراهيم",
    15: "الحجر",
    16: "النحل",
    17: "الإسراء",
    18: "الكهف",
    19: "مريم",
    20: "طه",
    21: "الأنبياء",
    22: "الحج",
    23: "المؤمنون",
    24: "النور",
    25: "الفرقان",
    26: "الشعراء",
    27: "النمل",
    28: "القصص",
    29: "العنكبوت",
    30: "الروم",
    31: "لقمان",
    32: "السجدة",
    33: "الأحزاب",
    34: "سبأ",
    35: "فاطر",
    36: "يس",
    37: "الصافات",
    38: "ص",
    39: "الزمر",
    40: "غافر",
    41: "فصلت",
    42: "الشورى",
    43: "الزخرف",
    44: "الدخان",
    45: "الجاثية",
    46: "الأحقاف",
    47: "محمد",
    48: "الفتح",
    49: "الحجرات",
    50: "ق",
    51: "الذاريات",
    52: "الطور",
    53: "النجم",
    54: "القمر",
    55: "الرحمن",
    56: "الواقعة",
    57: "الحديد",
    58: "المجادلة",
    59: "الحشر",
    60: "الممتحنة",
    61: "الصف",
    62: "الجمعة",
    63: "المنافقون",
    64: "التغابن",
    65: "الطلاق",
    66: "التحريم",
    67: "الملك",
    68: "القلم",
    69: "الحاقة",
    70: "المعارج",
    71: "نوح",
    72: "الجن",
    73: "المزمل",
    74: "المدثر",
    75: "القيامة",
    76: "الإنسان",
    77: "المرسلات",
    78: "النبأ",
    79: "النازعات",
    80: "عبس",
    81: "التكوير",
    82: "الانفطار",
    83: "المطففين",
    84: "الانشقاق",
    85: "البروج",
    86: "الطارق",
    87: "الأعلى",
    88: "الغاشية",
    89: "الفجر",
    90: "البلد",
    91: "الشمس",
    92: "الليل",
    93: "الضحى",
    94: "الشرح",
    95: "التين",
    96: "العلق",
    97: "القدر",
    98: "البينة",
    99: "الزلزلة",
    100: "العاديات",
    101: "القارعة",
    102: "التكاثر",
    103: "العصر",
    104: "الهمزة",
    105: "الفيل",
    106: "قريش",
    107: "الماعون",
    108: "الكوثر",
    109: "الكافرون",
    110: "النصر",
    111: "المسد",
    112: "الإخلاص",
    113: "الفلق",
    114: "الناس",
}

# Total ayah count per surah
SURAH_AYAH_COUNTS: dict[int, int] = {
    1: 7,
    2: 285,
    3: 200,
    4: 175,
    5: 122,
    6: 167,
    7: 206,
    8: 76,
    9: 130,
    10: 109,
    11: 121,
    12: 111,
    13: 44,
    14: 54,
    15: 99,
    16: 128,
    17: 110,
    18: 105,
    19: 99,
    20: 134,
    21: 111,
    22: 76,
    23: 119,
    24: 62,
    25: 77,
    26: 226,
    27: 95,
    28: 88,
    29: 69,
    30: 59,
    31: 33,
    32: 30,
    33: 73,
    34: 54,
    35: 46,
    36: 82,
    37: 182,
    38: 86,
    39: 72,
    40: 84,
    41: 53,
    42: 50,
    43: 89,
    44: 56,
    45: 36,
    46: 34,
    47: 39,
    48: 29,
    49: 18,
    50: 45,
    51: 60,
    52: 47,
    53: 61,
    54: 55,
    55: 77,
    56: 99,
    57: 28,
    58: 21,
    59: 24,
    60: 13,
    61: 14,
    62: 11,
    63: 11,
    64: 18,
    65: 12,
    66: 12,
    67: 31,
    68: 52,
    69: 52,
    70: 44,
    71: 30,
    72: 28,
    73: 18,
    74: 55,
    75: 39,
    76: 31,
    77: 50,
    78: 40,
    79: 45,
    80: 42,
    81: 29,
    82: 19,
    83: 36,
    84: 25,
    85: 22,
    86: 17,
    87: 19,
    88: 26,
    89: 32,
    90: 20,
    91: 15,
    92: 21,
    93: 11,
    94: 8,
    95: 8,
    96: 20,
    97: 5,
    98: 8,
    99: 9,
    100: 11,
    101: 10,
    102: 8,
    103: 3,
    104: 9,
    105: 5,
    106: 5,
    107: 6,
    108: 3,
    109: 6,
    110: 3,
    111: 5,
    112: 4,
    113: 5,
    114: 6,
}


class Surah(BaseModel):
    """
    Represents a Surah (chapter) of the Quran.

    Attributes:
        id: Surah number (1-114)
        name_arabic: Arabic name of the surah
        name_english: English transliteration (optional)
        total_ayahs: Total number of ayahs in this surah
        revelation_type: Makki or Madani (optional)
    """

    id: int = Field(
        ...,
        description="Surah number (1-114)",
        ge=1,
        le=114,
    )
    name_arabic: str = Field(
        ...,
        description="Arabic name of the surah",
    )
    name_english: Optional[str] = Field(
        default=None,
        description="English transliteration of the surah name",
    )
    total_ayahs: int = Field(
        ...,
        description="Total number of ayahs in this surah",
        ge=1,
    )
    revelation_type: Optional[str] = Field(
        default=None,
        description="Revelation type: 'makki' or 'madani'",
    )

    @classmethod
    def from_id(cls, surah_id: int) -> "Surah":
        """
        Create a Surah instance from its ID using built-in metadata.

        Args:
            surah_id: Surah number (1-114)

        Returns:
            Surah instance with metadata
        """
        if surah_id < 1 or surah_id > 114:
            raise ValueError(f"Invalid surah_id: {surah_id}. Must be 1-114.")

        return cls(
            id=surah_id,
            name_arabic=SURAH_NAMES[surah_id],
            total_ayahs=SURAH_AYAH_COUNTS[surah_id],
        )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": 1,
                    "name_arabic": "الفاتحة",
                    "name_english": "Al-Fatiha",
                    "total_ayahs": 7,
                    "revelation_type": "makki",
                }
            ]
        }
    }

    def __str__(self) -> str:
        return f"Surah {self.id}: {self.name_arabic}"
