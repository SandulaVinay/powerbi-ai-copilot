import re


class QueryRouter:

    # ========================================================
    # POWER BI / BUSINESS INTELLIGENCE DOMAIN
    # ========================================================

    POWER_BI_TERMS = [

        # ----------------------------------------------------
        # Power BI
        # ----------------------------------------------------

        r"\bpower\s*bi\b",
        r"\bpbi\b",

        # ----------------------------------------------------
        # Microsoft BI ecosystem
        # ----------------------------------------------------

        r"\bmicrosoft\s*fabric\b",
        r"\bfabric\b",
        r"\bpower\s*query\b",
        r"\bpower\s*bi\s*desktop\b",
        r"\bpower\s*bi\s*service\b",

        # ----------------------------------------------------
        # DAX / semantic modeling
        # ----------------------------------------------------

        r"\bdax\b",
        r"\bsemantic\s*model\b",
        r"\bsemantic\s*models\b",
        r"\bdata\s*model\b",
        r"\bdata\s*modeling\b",
        r"\bcalculated\s*column\b",
        r"\bcalculated\s*table\b",
        r"\bmeasure\b",
        r"\bmeasures\b",
        r"\btmdl\b",

        # ----------------------------------------------------
        # Connectivity / refresh
        # ----------------------------------------------------

        r"\bdirectquery\b",
        r"\bimport\s*mode\b",
        r"\blive\s*connection\b",
        r"\bincremental\s*refresh\b",
        r"\bdata\s*refresh\b",
        r"\bquery\s*folding\b",
        r"\bgateway\b",
        r"\bon-premises\s*gateway\b",

        # ----------------------------------------------------
        # Power BI features
        # ----------------------------------------------------

        r"\brls\b",
        r"\brow[-\s]*level\s*security\b",
        r"\bvisual\b",
        r"\bvisuals\b",
        r"\bdashboard\b",
        r"\bdashboards\b",
        r"\breport\b",
        r"\breports\b",
        r"\bworkspace\b",
        r"\bworkspaces\b",
        r"\bdataflow\b",
        r"\bdataflows\b",
        r"\bdata\s*mart\b",
        r"\bpaginated\s*report\b",
        r"\bpaginated\s*reports\b",
        r"\bcopilot\b",

        # ----------------------------------------------------
        # BI concepts
        # ----------------------------------------------------

        r"\bbusiness\s*intelligence\b",
        r"\bbusiness\s*analytics\b",
        r"\bbi\s*platform\b",
        r"\bbi\s*tool\b",
        r"\bbi\s*tools\b",
        r"\banalytics\b",
        r"\bdata\s*visualization\b",
        r"\bdata\s*warehouse\b",
        r"\betl\b",
        r"\belt\b",

        # ----------------------------------------------------
        # Related BI technologies
        # ----------------------------------------------------

        r"\btableau\b",
        r"\bqlik\b",
        r"\bqlik\s*sense\b",
        r"\blooker\b",
        r"\blooker\s*studio\b",
        r"\bexcel\b",
        r"\bssrs\b",
        r"\bsql\s*server\b",
        r"\bsynapse\b",
        r"\bazure\s*analysis\s*services\b",
        r"\baas\b",

    ]

    # ========================================================
    # WEB / CURRENT INFORMATION PATTERNS
    # ========================================================
    #
    # These patterns intentionally cover:
    #
    # 1. Current information
    # 2. Product updates
    # 3. Historical/discovery questions
    # 4. Comparisons
    #
    # ========================================================

    WEB_PATTERNS = [

        # ----------------------------------------------------
        # Current / freshness
        # ----------------------------------------------------

        r"\blatest\b",
        r"\bcurrent\b",
        r"\bcurrently\b",
        r"\bright\s+now\b",
        r"\btoday\b",
        r"\btonight\b",
        r"\bthis\s+week\b",
        r"\bthis\s+month\b",
        r"\bthis\s+year\b",

        r"\brecent\b",
        r"\brecently\b",
        r"\bnewest\b",
        r"\bmost\s+recent\b",

        # ----------------------------------------------------
        # What's new / updates
        # ----------------------------------------------------

        r"\bwhat'?s\s+new\b",
        r"\bwhat\s+is\s+new\b",
        r"\bwhat\s+are\s+the\s+new\b",

        r"\bnew\s+update\b",
        r"\bnew\s+updates\b",
        r"\blatest\s+update\b",
        r"\blatest\s+updates\b",

        r"\bnew\s+feature\b",
        r"\bnew\s+features\b",
        r"\blatest\s+feature\b",
        r"\blatest\s+features\b",

        r"\bwhat\s+changed\b",
        r"\bwhat\s+has\s+changed\b",
        r"\bchanges\b",
        r"\bupdates\b",
        r"\bupdate\b",

        # ----------------------------------------------------
        # Release / announcement information
        # ----------------------------------------------------

        r"\brelease\s+date\b",
        r"\brelease\b",
        r"\breleased\b",
        r"\bintroduced\b",
        r"\bannounced\b",
        r"\bannouncement\b",
        r"\blaunch(?:ed)?\b",
        r"\bpreview\b",
        r"\bpublic\s+preview\b",
        r"\bdeprecat(?:ed|ion|e)\b",
        r"\bretired\b",

        # ----------------------------------------------------
        # Historical / discovery questions
        # ----------------------------------------------------

        r"\bwho\s+invented\b",
        r"\bwho\s+created\b",
        r"\bwho\s+developed\b",
        r"\bwho\s+founded\b",
        r"\bwho\s+built\b",
        r"\bwho\s+made\b",
        r"\bwho\s+was\s+behind\b",
        r"\bcreator\s+of\b",
        r"\bcreators\s+of\b",
        r"\bdeveloper\s+of\b",
        r"\bdevelopers\s+of\b",
        r"\bhistory\s+of\b",
        r"\borigin\s+of\b",

        r"\bwhen\s+was\b",
        r"\bwhen\s+did\b",
        r"\bwhen\s+was\s+it\s+created\b",
        r"\bwhen\s+was\s+it\s+released\b",

        # ----------------------------------------------------
        # Comparisons
        # ----------------------------------------------------

        r"\bvs\.?\b",
        r"\bversus\b",
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bcomparisons\b",
        r"\bdifference\s+between\b",
        r"\bdifferences\s+between\b",
        r"\bdiffer\s+from\b",
        r"\bbetter\s+than\b",
        r"\bwhich\s+is\s+better\b",
        r"\bpros\s+and\s+cons\b",

    ]

    # ========================================================
    # EXPLICITLY OFF-TOPIC PATTERNS
    # ========================================================

    OFF_TOPIC_PATTERNS = [

        # ----------------------------------------------------
        # Movies / entertainment
        # ----------------------------------------------------

        r"\bmovie\b",
        r"\bmovies\b",
        r"\bfilm\b",
        r"\bfilms\b",
        r"\bnetflix\b",
        r"\bprime\s*video\b",
        r"\bspotify\b",
        r"\bsong\b",
        r"\bsongs\b",
        r"\bmusic\b",

        # ----------------------------------------------------
        # Sports
        # ----------------------------------------------------

        r"\bfootball\b",
        r"\bsoccer\b",
        r"\bcricket\b",
        r"\bnba\b",
        r"\bnfl\b",
        r"\btennis\b",
        r"\bbaseball\b",
        r"\bhockey\b",
        r"\bgolf\b",

        # ----------------------------------------------------
        # Weather
        # ----------------------------------------------------

        r"\bweather\b",
        r"\btemperature\b",
        r"\bforecast\b",

        # ----------------------------------------------------
        # Lifestyle
        # ----------------------------------------------------

        r"\brecipe\b",
        r"\brecipes\b",
        r"\brestaurant\b",
        r"\brestaurants\b",
        r"\btravel\b",
        r"\bvacation\b",
        r"\bhotel\b",
        r"\bhotels\b",

        # ----------------------------------------------------
        # Programming unrelated to BI
        # ----------------------------------------------------

        r"\bwrite\s+me\s+a\s+game\b",
        r"\bmake\s+a\s+game\b",
        r"\bwrite\s+python\s+code\b",
        r"\bwrite\s+javascript\b",
        r"\bwrite\s+java\s+code\b",
        r"\bprogram\s+a\s+game\b",

    ]

    # ========================================================
    # DOMAIN DETECTION
    # ========================================================

    @classmethod
    def is_power_bi_domain(
        cls,
        question: str
    ) -> bool:

        question = (
            str(question)
            .lower()
            .strip()
        )

        for pattern in cls.POWER_BI_TERMS:

            if re.search(
                pattern,
                question
            ):
                return True

        return False

    # ========================================================
    # EXPLICIT OFF-TOPIC DETECTION
    # ========================================================

    @classmethod
    def is_off_topic(
        cls,
        question: str
    ) -> bool:

        question = (
            str(question)
            .lower()
            .strip()
        )

        for pattern in cls.OFF_TOPIC_PATTERNS:

            if re.search(
                pattern,
                question
            ):
                return True

        return False

    # ========================================================
    # CURRENT / WEB QUESTION
    # ========================================================

    @classmethod
    def requires_web(
        cls,
        question: str
    ) -> bool:

        question = (
            str(question)
            .lower()
            .strip()
        )

        for pattern in cls.WEB_PATTERNS:

            if re.search(
                pattern,
                question
            ):
                return True

        return False

    # ========================================================
    # MAIN CLASSIFIER
    #
    # Returns:
    #
    # "local"
    # "web"
    # "reject"
    #
    # ========================================================

    @classmethod
    def classify(
        cls,
        question: str
    ) -> str:

        question = (
            str(question)
            .lower()
            .strip()
        )

        # ====================================================
        # 1. EMPTY QUESTION
        # ====================================================

        if not question:
            return "reject"

        # ====================================================
        # 2. EXPLICIT OFF-TOPIC
        #
        # Important:
        #
        # "How can I create a Power BI dashboard
        # for movies?"
        #
        # contains "movie", but it is still a Power BI
        # question.
        # ====================================================

        if cls.is_off_topic(question):

            if not cls.is_power_bi_domain(question):
                return "reject"

        # ====================================================
        # 3. EXPLICIT POWER BI / BI DOMAIN
        # ====================================================

        if cls.is_power_bi_domain(question):

            # Current, historical and comparison questions
            # should use live web retrieval.

            if cls.requires_web(question):
                return "web"

            return "local"

        # ====================================================
        # 4. IMPLICIT BI QUESTIONS
        #
        # These are useful when the user doesn't explicitly
        # write "Power BI" but clearly uses BI terminology.
        # ====================================================

        implicit_bi_patterns = [

            r"\bsemantic\b",
            r"\bdata\s*model\b",
            r"\bdata\s*modeling\b",
            r"\bdata\s*visualization\b",
            r"\bbusiness\s*analytics\b",
            r"\bbusiness\s*intelligence\b",
            r"\bbi\b",
            r"\bdax\b",
            r"\bmeasure\b",
            r"\bmeasures\b",
            r"\bquery\s*folding\b",
            r"\bincremental\s*refresh\b",
            r"\bdirectquery\b",
            r"\bdata\s*refresh\b",

        ]

        for pattern in implicit_bi_patterns:

            if re.search(
                pattern,
                question
            ):

                if cls.requires_web(question):
                    return "web"

                return "local"

        # ====================================================
        # 5. EVERYTHING ELSE
        # ====================================================

        return "reject"

    # ========================================================
    # HUMAN-READABLE REJECTION MESSAGE
    # ========================================================

    @staticmethod
    def rejection_message():

        return (
            "I'm a Power BI-focused assistant. "
            "I can help with Power BI, Microsoft Fabric, "
            "Power Query, DAX, semantic models, "
            "Power BI Service, Power BI Desktop, "
            "business intelligence, analytics, "
            "performance, administration, and related "
            "BI technologies."
        )