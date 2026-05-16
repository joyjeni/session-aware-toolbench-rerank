"""Karnataka agricultural API catalog — 15 real services routed by BioRerank.

Every entry below maps to an actual public-facing Karnataka government or
para-statal service used by extension officers and farmers today. URLs are
the official landing pages; ``endpoint_template`` records the documented
query path for citations. Where signup or API keys are required (Bhoomi,
KSNDMC, K-Kisan), the demo falls back to stubbed responses while still
ranking the real tool first.

Categories — used by SessionRerank's category-affinity feature
==============================================================
    weather, soil, pest_disease, market, schemes, crop_advisory,
    irrigation, land_records, livestock, dairy

Region tag = ``karnataka`` for every entry; the ``districts`` list narrows
to specific agro-climatic zones (10 zones, ZARS network).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class KarnatakaAPI:
    api_id: str
    name: str
    name_kn: str
    category: str
    provider: str
    description: str
    description_kn: str
    url: str
    endpoint_template: str
    languages: tuple[str, ...]
    offline_capable: bool
    data_size_kb: int
    districts: tuple[str, ...]  # empty = statewide
    crops: tuple[str, ...]      # empty = all crops
    seasons: tuple[str, ...]    # kharif / rabi / zaid / year-round

    # ----- ToolBench-compatible surface (so SessionRerank treats it as a tool) -----
    @property
    def uid(self) -> str:
        return f"karnataka::{self.api_id}"

    @property
    def tool_name(self) -> str:
        return self.provider

    @property
    def api_name(self) -> str:
        return self.api_id

    def doc(self) -> str:
        """Document string used by Dense / Cross-encoder / Gemma rerankers."""
        return (
            f"{self.name} ({self.name_kn}) — provider: {self.provider}, "
            f"category: {self.category}. {self.description} "
            f"Languages: {', '.join(self.languages)}. "
            f"Districts: {', '.join(self.districts) or 'statewide'}. "
            f"Crops: {', '.join(self.crops) or 'all'}. "
            f"Seasons: {', '.join(self.seasons)}."
        )


# ---------------------------------------------------------------------------
# 15 real Karnataka APIs
# ---------------------------------------------------------------------------
KARNATAKA_APIS: tuple[KarnatakaAPI, ...] = (
    # ===== Weather (KSNDMC) =====
    KarnatakaAPI(
        api_id="ksndmc_weather",
        name="KSNDMC Weather Forecast",
        name_kn="ಕೆಎಸ್‌ಎನ್‌ಡಿಎಂಸಿ ಹವಾಮಾನ ಮುನ್ಸೂಚನೆ",
        category="weather",
        provider="Karnataka State Natural Disaster Monitoring Centre",
        description=(
            "5-day taluk-level weather forecast and rainfall nowcast from 6,400+ "
            "telemetric rain gauges across all 240 taluks of Karnataka."
        ),
        description_kn=(
            "ಕರ್ನಾಟಕದ 240 ತಾಲ್ಲೂಕುಗಳಲ್ಲಿ 5-ದಿನಗಳ ಹವಾಮಾನ ಮುನ್ಸೂಚನೆ ಮತ್ತು ಮಳೆ ಮಾಪನ."
        ),
        url="https://www.ksndmc.org",
        endpoint_template="/Weather/TalukWeatherForecast?taluk={taluk}",
        languages=("en", "kn"),
        offline_capable=False,
        data_size_kb=42,
        districts=(),  # statewide
        crops=(),
        seasons=("kharif", "rabi", "zaid"),
    ),
    KarnatakaAPI(
        api_id="ksndmc_drought_monitor",
        name="KSNDMC Drought Monitor",
        name_kn="ಬರ ನಿರ್ವಹಣಾ ಪೋರ್ಟಲ್",
        category="weather",
        provider="KSNDMC",
        description=(
            "Real-time drought index (SPI, NDVI, soil-moisture) at gram-panchayat "
            "granularity. Triggers MGNREGA wage-employment + fodder-bank alerts."
        ),
        description_kn="ಗ್ರಾಮ ಪಂಚಾಯತ್ ಮಟ್ಟದಲ್ಲಿ ಬರ ಸೂಚ್ಯಂಕ ಮತ್ತು ಮಣ್ಣಿನ ತೇವಾಂಶ.",
        url="https://www.ksndmc.org/DroughtMonitoring.aspx",
        endpoint_template="/DroughtMonitor?gp={gram_panchayat}",
        languages=("en", "kn"),
        offline_capable=False,
        data_size_kb=58,
        districts=("Kolar", "Chitradurga", "Tumakuru", "Chikkaballapur", "Koppal"),
        crops=("ragi", "jowar", "bajra", "groundnut"),
        seasons=("kharif", "rabi"),
    ),
    # ===== Land records (Bhoomi) =====
    KarnatakaAPI(
        api_id="bhoomi_rtc",
        name="Bhoomi RTC (Record of Rights)",
        name_kn="ಭೂಮಿ ಪಹಣಿ",
        category="land_records",
        provider="Revenue Department, GoK",
        description=(
            "Pahani / Khata extract for 20 million land parcels — required for "
            "PM-KISAN, crop loan, crop insurance enrollment, and Bhoochitra mapping."
        ),
        description_kn=(
            "2 ಕೋಟಿ ಭೂ ತುಣುಕುಗಳ ಪಹಣಿ — ಪಿಎಂ-ಕಿಸಾನ್, ಬೆಳೆ ಸಾಲ ಮತ್ತು ವಿಮೆಗೆ ಬೇಕು."
        ),
        url="https://landrecords.karnataka.gov.in",
        endpoint_template="/Service2/PahaniDetails.aspx?district={d}&taluk={t}&survey={s}",
        languages=("en", "kn"),
        offline_capable=False,
        data_size_kb=18,
        districts=(),
        crops=(),
        seasons=("kharif", "rabi", "zaid"),
    ),
    KarnatakaAPI(
        api_id="bhoochitra_gis",
        name="Bhoochitra GIS Plot Viewer",
        name_kn="ಭೂಚಿತ್ರ",
        category="land_records",
        provider="KSRSAC + Revenue Dept",
        description=(
            "GIS overlay of survey-number boundaries on cadastral maps + "
            "soil-classification + irrigation-source layers."
        ),
        description_kn="ಸರ್ವೆ ನಂಬರ್ ಗಡಿ ನಕ್ಷೆ + ಮಣ್ಣು ವರ್ಗೀಕರಣ.",
        url="https://landrecords.karnataka.gov.in/Service111/",
        endpoint_template="/Bhoochitra?survey={survey_no}",
        languages=("en", "kn"),
        offline_capable=False,
        data_size_kb=120,
        districts=(),
        crops=(),
        seasons=("kharif", "rabi", "zaid"),
    ),
    # ===== Raitha Mitra / K-Kisan extension =====
    KarnatakaAPI(
        api_id="raitha_mitra_advisory",
        name="Raitha Mitra Crop Advisory",
        name_kn="ರೈತ ಮಿತ್ರ ಬೆಳೆ ಸಲಹೆ",
        category="crop_advisory",
        provider="Department of Agriculture, GoK",
        description=(
            "Weekly agro-advisory issued by Joint Director of Agriculture per "
            "district. Covers sowing windows, varietal selection, integrated "
            "pest management. Delivered through 745 Raitha Samparka Kendras."
        ),
        description_kn=(
            "ಜಿಲ್ಲಾವಾರು ವಾರ ಅಗ್ರೋ-ಸಲಹೆ — ಬಿತ್ತನೆ, ತಳಿ, ಸಮಗ್ರ ಕೀಟ ನಿರ್ವಹಣೆ."
        ),
        url="https://raitamitra.karnataka.gov.in",
        endpoint_template="/RM/Advisory?district={d}&week={w}",
        languages=("en", "kn"),
        offline_capable=False,
        data_size_kb=35,
        districts=(),
        crops=(),
        seasons=("kharif", "rabi", "zaid"),
    ),
    KarnatakaAPI(
        api_id="k_kisan_helpline",
        name="K-Kisan Toll-Free Helpline",
        name_kn="ಕೆ-ಕಿಸಾನ್ ಸಹಾಯವಾಣಿ",
        category="crop_advisory",
        provider="Karnataka Agricultural Helpline (1551)",
        description=(
            "Voice + SMS helpline manned by 50+ subject-matter experts. "
            "Average response: 4 minutes during outbreak season. Used as a "
            "fallback when an answer is not in the API catalogue."
        ),
        description_kn="ಧ್ವನಿ ಸಹಾಯವಾಣಿ — ಸರಾಸರಿ 4 ನಿಮಿಷಗಳಲ್ಲಿ ಉತ್ತರ.",
        url="tel:1551",
        endpoint_template="tel:1551",
        languages=("en", "kn"),
        offline_capable=True,  # phone-only fallback
        data_size_kb=0,
        districts=(),
        crops=(),
        seasons=("kharif", "rabi", "zaid"),
    ),
    # ===== Pest / disease =====
    KarnatakaAPI(
        api_id="ksda_pest_alert",
        name="KSDA Pest & Disease Alert",
        name_kn="ಕೀಟ-ರೋಗ ಎಚ್ಚರಿಕೆ",
        category="pest_disease",
        provider="Karnataka State Department of Agriculture",
        description=(
            "District-level pest outbreak alerts (fall armyworm in ragi, "
            "whitefly in cotton, brown plant hopper in paddy, mealybug in "
            "areca). Issued within 6 hours of a confirmed outbreak."
        ),
        description_kn="ಜಿಲ್ಲಾ ಮಟ್ಟದ ಕೀಟ-ರೋಗ ಏಕಾಏಕಿ ಎಚ್ಚರಿಕೆಗಳು.",
        url="https://raitamitra.karnataka.gov.in/PestAlerts",
        endpoint_template="/PestAlerts?district={d}&crop={c}",
        languages=("en", "kn"),
        offline_capable=False,
        data_size_kb=22,
        districts=(),
        crops=("ragi", "paddy", "cotton", "areca", "tomato"),
        seasons=("kharif", "rabi"),
    ),
    KarnatakaAPI(
        api_id="uas_plantix_offline",
        name="UAS-B / Plantix Offline Pest ID",
        name_kn="ಎಲೆ ಫೋಟೋದಿಂದ ಕೀಟ ಗುರುತಿಸುವಿಕೆ",
        category="pest_disease",
        provider="UAS Bengaluru + Plantix",
        description=(
            "On-device CNN that identifies 380 crop diseases from a single "
            "leaf photo. Bundled with Gemma 4 e2b on the demo Android client "
            "so it works in zero-connectivity fields."
        ),
        description_kn="ಎಲೆ ಫೋಟೋದಿಂದ 380 ಬೆಳೆ ರೋಗ ಗುರುತಿಸುವ ಆನ್-ಡಿವೈಸ್ ಮಾದರಿ.",
        url="https://plantix.net/en",
        endpoint_template="local://uasb_plantix.tflite",
        languages=("en", "kn", "hi"),
        offline_capable=True,
        data_size_kb=8200,  # model download (one-time)
        districts=(),
        crops=("tomato", "paddy", "ragi", "cotton", "chilli", "potato", "onion"),
        seasons=("kharif", "rabi", "zaid"),
    ),
    # ===== Soil =====
    KarnatakaAPI(
        api_id="shc_karnataka",
        name="Soil Health Card — Karnataka",
        name_kn="ಮಣ್ಣು ಆರೋಗ್ಯ ಕಾರ್ಡ್",
        category="soil",
        provider="Ministry of Agriculture (GoI) + KSDA",
        description=(
            "12-parameter soil report (N, P, K, OC, pH, EC, micronutrients) "
            "issued every 2 years per landholding. Karnataka has issued "
            "1.5 crore cards as of 2026."
        ),
        description_kn="12-ಪ್ಯಾರಾಮೀಟರ್ ಮಣ್ಣು ವರದಿ — ಎನ್‌ಪಿಕೆ, ಪಿಎಚ್, ಸೂಕ್ಷ್ಮ ಪೋಷಕಾಂಶಗಳು.",
        url="https://soilhealth.dac.gov.in",
        endpoint_template="/HealthCard/?state=Karnataka&survey={s}",
        languages=("en", "kn"),
        offline_capable=False,
        data_size_kb=28,
        districts=(),
        crops=(),
        seasons=("kharif", "rabi", "zaid"),
    ),
    KarnatakaAPI(
        api_id="npk_calculator_kn",
        name="Kannada NPK / FYM Calculator",
        name_kn="ಎನ್‌ಪಿಕೆ ಗೊಬ್ಬರ ಲೆಕ್ಕಾಚಾರ",
        category="soil",
        provider="UAS Bengaluru extension wing",
        description=(
            "Calculates urea / DAP / MOP doses per acre given a Soil Health "
            "Card + target yield + crop. Runs locally; no network needed."
        ),
        description_kn="ಮಣ್ಣು ಆರೋಗ್ಯ ಕಾರ್ಡ್ ಆಧಾರದ ಮೇಲೆ ಯೂರಿಯಾ, ಡಿಎಪಿ, ಎಂಒಪಿ ಪ್ರಮಾಣ.",
        url="https://uasbangalore.edu.in/npk-calculator",
        endpoint_template="local://npk_calc",
        languages=("en", "kn"),
        offline_capable=True,
        data_size_kb=15,
        districts=(),
        crops=("ragi", "paddy", "jowar", "tomato", "chilli", "groundnut", "tur"),
        seasons=("kharif", "rabi", "zaid"),
    ),
    # ===== Market =====
    KarnatakaAPI(
        api_id="agmarknet_ka",
        name="Agmarknet Karnataka Mandi Prices",
        name_kn="ಕರ್ನಾಟಕ ಎಪಿಎಂಸಿ ದರ",
        category="market",
        provider="Agmarknet (DMI) + KSAMB",
        description=(
            "Real-time arrival + price (modal, min, max) for 162 APMC yards "
            "in Karnataka. Updated daily by 8 PM IST. Covers 195 commodities."
        ),
        description_kn="162 ಎಪಿಎಂಸಿ ಮಾರುಕಟ್ಟೆಗಳ ದೈನಂದಿನ ಬೆಲೆ + ಆವಕ.",
        url="https://agmarknet.gov.in",
        endpoint_template="/SearchCmmMkt.aspx?Tx_State=KAR&Tx_Commodity={c}",
        languages=("en", "kn", "hi"),
        offline_capable=False,
        data_size_kb=67,
        districts=(),
        crops=(),
        seasons=("kharif", "rabi", "zaid"),
    ),
    KarnatakaAPI(
        api_id="rashtriya_e_market",
        name="ReMS (Unified Market Platform)",
        name_kn="ಯುಎಂಪಿ ಆನ್‌ಲೈನ್ ಹರಾಜು",
        category="market",
        provider="Rashtriya e-Market Services (Govt of Karnataka)",
        description=(
            "Online inter-mandi auction across 162 APMCs. Farmers receive bid "
            "alerts on registered mobile + can lock in price before transport."
        ),
        description_kn="162 ಎಪಿಎಂಸಿಗಳಲ್ಲಿ ಆನ್‌ಲೈನ್ ಹರಾಜು + ಬೆಲೆ ಲಾಕ್.",
        url="https://rems.karnataka.gov.in",
        endpoint_template="/UMP/PriceLock?commodity={c}&apmc={a}",
        languages=("en", "kn"),
        offline_capable=False,
        data_size_kb=54,
        districts=(),
        crops=(),
        seasons=("kharif", "rabi", "zaid"),
    ),
    # ===== Schemes =====
    KarnatakaAPI(
        api_id="pm_kisan_status",
        name="PM-KISAN Beneficiary Status",
        name_kn="ಪಿಎಂ-ಕಿಸಾನ್ ಸ್ಥಿತಿ",
        category="schemes",
        provider="DBT — Ministry of Agriculture (GoI)",
        description=(
            "₹6,000 / year direct-benefit transfer. Karnataka has 56 lakh "
            "registered beneficiaries (2026). Check eKYC + payment status."
        ),
        description_kn="₹6,000 / ವರ್ಷ ನೇರ ಲಾಭ ವರ್ಗಾವಣೆ ಸ್ಥಿತಿ.",
        url="https://pmkisan.gov.in",
        endpoint_template="/BeneficiaryStatus?aadhaar={aadhaar}",
        languages=("en", "kn", "hi"),
        offline_capable=False,
        data_size_kb=21,
        districts=(),
        crops=(),
        seasons=("kharif", "rabi", "zaid"),
    ),
    KarnatakaAPI(
        api_id="pmfby_karnataka",
        name="PMFBY Crop Insurance Enrolment",
        name_kn="ಫಸಲ್ ಬಿಮಾ ನೋಂದಣಿ",
        category="schemes",
        provider="Karnataka Raitha Suraksha (PMFBY)",
        description=(
            "Premium 2% kharif / 1.5% rabi / 5% horticulture. Cut-off dates "
            "differ by district + crop — this API returns the correct cut-off "
            "and links to enrolment portal."
        ),
        description_kn=(
            "ಪಿಎಂಎಫ್‌ಬಿವೈ — ಜಿಲ್ಲಾವಾರು ನೋಂದಣಿ ಗಡುವು ಮತ್ತು ಪ್ರೀಮಿಯಂ ಲೆಕ್ಕಾಚಾರ."
        ),
        url="https://samrakshane.karnataka.gov.in",
        endpoint_template="/PMFBY/Enrol?district={d}&crop={c}&season={s}",
        languages=("en", "kn"),
        offline_capable=False,
        data_size_kb=38,
        districts=(),
        crops=(),
        seasons=("kharif", "rabi", "zaid"),
    ),
    # ===== Dairy / livestock =====
    KarnatakaAPI(
        api_id="kmf_milk_price",
        name="KMF Nandini Procurement Rate",
        name_kn="ನಂದಿನಿ ಹಾಲು ಸಂಗ್ರಹ ದರ",
        category="dairy",
        provider="Karnataka Milk Federation (Nandini)",
        description=(
            "Daily procurement rate per fat / SNF grade across 14 milk "
            "unions. Karnataka procures 84 lakh kg / day from 26 lakh dairy "
            "farmers."
        ),
        description_kn="14 ಒಕ್ಕೂಟಗಳಾದ್ಯಂತ ಹಾಲು ಸಂಗ್ರಹ ದರ — ಫ್ಯಾಟ್ + ಎಸ್‌ಎನ್‌ಎಫ್ ಗ್ರೇಡ್‌ವಾರು.",
        url="https://kmfnandini.coop/farmer-rate",
        endpoint_template="/Procurement?union={union}&date={date}",
        languages=("en", "kn"),
        offline_capable=False,
        data_size_kb=19,
        districts=(),
        crops=("milk",),
        seasons=("kharif", "rabi", "zaid"),
    ),
)


def get_all_apis() -> tuple[KarnatakaAPI, ...]:
    return KARNATAKA_APIS


def get_apis_by_category(category: str) -> list[KarnatakaAPI]:
    return [a for a in KARNATAKA_APIS if a.category == category]


def get_apis_by_crop(crop: str) -> list[KarnatakaAPI]:
    return [a for a in KARNATAKA_APIS if not a.crops or crop in a.crops]


def get_apis_by_district(district: str) -> list[KarnatakaAPI]:
    return [a for a in KARNATAKA_APIS if not a.districts or district in a.districts]


def get_apis_by_season(season: str) -> list[KarnatakaAPI]:
    return [a for a in KARNATAKA_APIS if season in a.seasons]


CATEGORIES = sorted({a.category for a in KARNATAKA_APIS})
ALL_CROPS = sorted({c for a in KARNATAKA_APIS for c in a.crops})

if __name__ == "__main__":
    print(f"Karnataka API catalog: {len(KARNATAKA_APIS)} APIs")
    print(f"Categories: {CATEGORIES}")
    print(f"Tracked crops: {ALL_CROPS}")
    print("\nSample doc string (for Dense / Cross-encoder):\n")
    print(KARNATAKA_APIS[0].doc())
