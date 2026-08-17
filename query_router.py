import re


class QueryRouter:

    POWER_BI_TERMS = [
        r"\bpower\s*bi\b", r"\bpbi\b", r"\bmicrosoft\s*fabric\b", r"\bfabric\b",
        r"\bpower\s*query\b", r"\bpower\s*bi\s*desktop\b", r"\bpower\s*bi\s*service\b",
        r"\bdax\b", r"\bsemantic\s*model\b", r"\bsemantic\s*models\b", r"\bdata\s*model\b",
        r"\bdata\s*modeling\b", r"\bcalculated\s*column\b", r"\bcalculated\s*table\b",
        r"\bmeasure\b", r"\bmeasures\b", r"\btmdl\b", r"\bdirectquery\b",
        r"\bimport\s*mode\b", r"\blive\s*connection\b", r"\bincremental\s*refresh\b",
        r"\bdata\s*refresh\b", r"\bquery\s*folding\b", r"\bgateway\b",
        r"\bon-premises\s*gateway\b", r"\brls\b", r"\brow[-\s]*level\s*security\b",
        r"\bvisual\b", r"\bvisuals\b", r"\bdashboard\b", r"\bdashboards\b",
        r"\breport\b", r"\breports\b", r"\bworkspace\b", r"\bworkspaces\b",
        r"\bdataflow\b", r"\bdataflows\b", r"\bdata\s*mart\b", r"\bpaginated\s*report\b",
        r"\bpaginated\s*reports\b", r"\bcopilot\b", r"\bbusiness\s*intelligence\b",
        r"\bbusiness\s*analytics\b", r"\bbi\s*platform\b", r"\bbi\s*tool\b",
        r"\bbi\s*tools\b", r"\banalytics\b", r"\bdata\s*visualization\b",
        r"\bdata\s*warehouse\b", r"\betl\b", r"\belt\b", r"\btableau\b", r"\bqlik\b",
        r"\bqlik\s*sense\b", r"\blooker\b", r"\blooker\s*studio\b", r"\bexcel\b",
        r"\bssrs\b", r"\bsql\s*server\b", r"\bsynapse\b", r"\bazure\s*analysis\s*services\b",
        r"\baas\b", r"\bchurn\b", r"\bcustomer\s+churn\b", r"\bretention\s+rate\b",
        r"\bcohort\s+analysis\b", r"\bkpi\b", r"\bkpi\s+card\b", r"\bkpi\s+cards\b",
    ]

    DAX_FUNCTIONS = {
        "IF", "SWITCH", "AND", "OR", "NOT", "TRUE", "FALSE", "COALESCE", "IFERROR",
        "SUM", "SUMX", "AVERAGE", "AVERAGEX", "MIN", "MINX", "MAX", "MAXX",
        "COUNT", "COUNTA", "COUNTAX", "COUNTBLANK", "COUNTROWS", "DISTINCTCOUNT",
        "CALCULATE", "CALCULATETABLE", "FILTER", "ALL", "ALLEXCEPT", "ALLSELECTED",
        "REMOVEFILTERS", "KEEPFILTERS", "TREATAS", "DATE", "DATEDIFF", "DATEVALUE",
        "DAY", "EDATE", "EOMONTH", "HOUR", "MINUTE", "MONTH", "NOW", "QUARTER",
        "SECOND", "TIME", "TIMEVALUE", "TODAY", "WEEKDAY", "WEEKNUM", "YEAR",
        "CLOSINGBALANCEMONTH", "CLOSINGBALANCEQUARTER", "CLOSINGBALANCEYEAR", "DATESBETWEEN",
        "DATESINPERIOD", "DATESMTD", "DATESQTD", "DATESYTD", "ENDOFMONTH", "ENDOFQUARTER",
        "ENDOFYEAR", "FIRSTDATE", "LASTDATE", "NEXTDAY", "NEXTMONTH", "NEXTQUARTER",
        "NEXTYEAR", "PREVIOUSDAY", "PREVIOUSMONTH", "PREVIOUSQUARTER", "PREVIOUSYEAR",
        "STARTOFMONTH", "STARTOFQUARTER", "STARTOFYEAR", "TOTALMTD", "TOTALQTD", "TOTALYTD",
        "CONCATENATE", "CONCATENATEX", "CONTAINSSTRING", "EXACT", "FIND", "FORMAT", "LEFT",
        "LEN", "LOWER", "MID", "REPLACE", "RIGHT", "SEARCH", "SUBSTITUTE", "TRIM", "UPPER",
        "VALUE", "ABS", "CEILING", "DIVIDE", "EXP", "FLOOR", "INT", "LN", "LOG", "MOD",
        "POWER", "ROUND", "ROUNDDOWN", "ROUNDUP", "SQRT", "TRUNC", "ADDCOLUMNS", "CROSSJOIN",
        "DISTINCT", "EXCEPT", "GENERATE", "GENERATESERIES", "GROUPBY", "INTERSECT", "SELECTCOLUMNS",
        "SUMMARIZE", "SUMMARIZECOLUMNS", "UNION", "VALUES", "CROSSFILTER", "RELATED", "RELATEDTABLE",
        "USERELATIONSHIP", "ISBLANK", "ISERROR", "ISEMPTY", "ISFILTERED", "ISINSCOPE", "ISNUMBER",
        "ISTEXT", "HASONEFILTER", "HASONEVALUE", "MEDIAN", "MEDIANX", "RANK", "RANKX", "STDEV.P",
        "STDEV.S", "VAR.P", "VAR.S"
    }

    DAX_INTENT_PATTERNS = [
        r"\bdifference\s+between\b", r"\bdifferences\s+between\b", r"\bvs\.?\b",
        r"\bversus\b", r"\bcompare\b", r"\bcomparison\b", r"\bcompare\s+between\b",
        r"\balternative\b", r"\balternatives\b", r"\balternate\b", r"\breplace\b",
        r"\breplacement\b", r"\bother\s+than\b", r"\binstead\s+of\b", r"\bwhich\s+should\s+i\s+use\b",
        r"\bwhen\s+should\s+i\s+use\b", r"\bwhat\s+should\s+i\s+use\b",
        r"\bwhat\s+is\b", r"\bexplain\b", r"\bhow\s+does\b", r"\bhow\s+to\b",
    ]

    KNOWLEDGE_GAP_WEB_PATTERNS = [
        r"\bhow\s+to\s+install\b", r"\binstall\b.*\bpower\s*bi\b",
        r"\bpower\s*bi\s+desktop\b.*\binstall\b", r"\bdate\s+function(s)?\b",
        r"\bdax\s+function(s)?\b", r"\bchurn\b", r"\bretention\b", r"\bcohort\b",
        r"\btmdl\b.*\bhow\b", r"\bhow\s+to\s+use\s+tmdl\b",
        r"\bpower\s*bi\s+certification\b", r"\bcertification\b.*\bpower\s*bi\b",
        r"\blearn\s+power\s*bi\b", r"\blearning\s+power\s*bi\b",
        r"\bpower\s*bi\s+(course|courses|training|tutorial|tutorials)\b",
        r"\b(youtube|videos?)\b.*\bpower\s*bi\b", r"\bwhat\s+is\s+power\s*bi\b",
        r"\bhow\s+does\s+power\s*bi\s+work\b",
    ]

    LOCAL_CORE_PATTERNS = [
        r"\bincremental\s+refresh\b", r"\bdirectquery\b", r"\bquery\s+folding\b",
        r"\bdata\s+refresh\b", r"\bsemantic\s+model\b", r"\bcalculated\s+column\b",
        r"\bcalculated\s+table\b", r"\bmeasure(s)?\b",
    ]

    WEB_PATTERNS = [
        r"\blatest\b", r"\bcurrent\b", r"\bcurrently\b", r"\bright\s+now\b",
        r"\btoday\b", r"\btonight\b", r"\bthis\s+week\b", r"\bthis\s+month\b",
        r"\bthis\s+year\b", r"\brecent\b", r"\brecently\b", r"\bnewest\b",
        r"\bmost\s+recent\b", r"\bwhat'?s\s+new\b", r"\bwhat\s+is\s+new\b",
        r"\bwhat\s+are\s+the\s+new\b", r"\bnew\s+update\b", r"\bnew\s+updates\b",
        r"\bnew\s+feature\b", r"\bnew\s+features\b", r"\blatest\s+feature\b",
        r"\blatest\s+features\b", r"\bwhat\s+changed\b", r"\bwhat\s+has\s+changed\b",
        r"\bchanges\b", r"\bupdates\b", r"\bupdate\b", r"\brelease\s+date\b",
        r"\brelease\b", r"\breleased\b", r"\bintroduced\b", r"\bannounced\b",
        r"\bannouncement\b", r"\blaunch(?:ed)?\b", r"\bpreview\b", r"\bpublic\s+preview\b",
        r"\bdeprecat(?:ed|ion|e)\b", r"\bretired\b", r"\bwho\s+invented\b",
        r"\bwho\s+created\b", r"\bwho\s+developed\b", r"\bwho\s+founded\b",
        r"\bwho\s+built\b", r"\bwho\s+made\b", r"\bwho\s+was\s+behind\b",
        r"\bcreator\s+of\b", r"\bcreators\s+of\b", r"\bdeveloper\s+of\b",
        r"\bdevelopers\s+of\b", r"\bhistory\s+of\b", r"\borigin\s+of\b",
        r"\bwhen\s+was\b", r"\bwhen\s+did\b", r"\bwhen\s+was\s+it\s+created\b",
        r"\bwhen\s+was\s+it\s+released\b", r"\bvs\.?\b", r"\bversus\b",
        r"\bcompare\b", r"\bcomparison\b", r"\bcomparisons\b",
        r"\bdifference\s+between\b", r"\bdifferences\s+between\b", r"\bdiffer\s+from\b",
        r"\bbetter\s+than\b", r"\bwhich\s+is\s+better\b", r"\bpros\s+and\s+cons\b",
    ]

    OFF_TOPIC_PATTERNS = [
        r"\bmovie\b", r"\bmovies\b", r"\bfilm\b", r"\bfilms\b", r"\bnetflix\b",
        r"\bprime\s*video\b", r"\bspotify\b", r"\bsong\b", r"\bsongs\b", r"\bmusic\b",
        r"\bfootball\b", r"\bsoccer\b", r"\bcricket\b", r"\bnba\b", r"\bnfl\b",
        r"\btennis\b", r"\bbaseball\b", r"\bhockey\b", r"\bgolf\b", r"\bweather\b",
        r"\btemperature\b", r"\bforecast\b", r"\brecipe\b", r"\brecipes\b",
        r"\brestaurant\b", r"\brestaurants\b", r"\btravel\b", r"\bvacation\b",
        r"\bhotel\b", r"\bhotels\b", r"\bwrite\s+me\s+a\s+game\b",
        r"\bmake\s+a\s+game\b", r"\bwrite\s+python\s+code\b", r"\bwrite\s+javascript\b",
        r"\bwrite\s+java\s+code\b", r"\bprogram\s+a\s+game\b",
    ]

    @classmethod
    def extract_dax_functions(cls, question):
        q = str(question).lower().strip()
        return sorted({
            fn.upper()
            for fn in cls.DAX_FUNCTIONS
            if re.search(r"\b" + re.escape(fn.lower()) + r"\b", q)
        })

    @classmethod
    def is_dax_function_question(cls, question):
        q = str(question).lower().strip()
        functions = cls.extract_dax_functions(q)
        if not functions:
            return False
        has_dax_signal = bool(re.search(r"\bdax\b|\bfunction(s)?\b|\bmeasure(s)?\b", q))
        has_question_signal = any(re.search(pattern, q) for pattern in cls.DAX_INTENT_PATTERNS)
        return has_dax_signal or has_question_signal

    @classmethod
    def is_dax_alternative_question(cls, question):
        q = str(question).lower().strip()
        functions = cls.extract_dax_functions(q)
        if not functions:
            return False
        return any(re.search(pattern, q) for pattern in [
            r"\balternative(s)?\b", r"\balternate\b", r"\bother\s+than\b",
            r"\binstead\s+of\b", r"\breplacement\b", r"\breplace\b",
            r"\bwhich\s+.*\buse\b", r"\bwhat\s+.*\buse\b"
        ])

    @classmethod
    def is_power_bi_domain(cls, question):
        question = str(question).lower().strip()
        return any(re.search(pattern, question) for pattern in cls.POWER_BI_TERMS)

    @classmethod
    def is_off_topic(cls, question):
        question = str(question).lower().strip()
        return any(re.search(pattern, question) for pattern in cls.OFF_TOPIC_PATTERNS)

    @classmethod
    def requires_web(cls, question):
        question = str(question).lower().strip()
        return any(re.search(pattern, question) for pattern in cls.WEB_PATTERNS)

    @classmethod
    def needs_web_for_knowledge_gap(cls, question):
        question = str(question).lower().strip()
        return any(re.search(pattern, question) for pattern in cls.KNOWLEDGE_GAP_WEB_PATTERNS)

    @classmethod
    def is_local_core(cls, question):
        question = str(question).lower().strip()
        return any(re.search(pattern, question) for pattern in cls.LOCAL_CORE_PATTERNS)

    @classmethod
    def classify(cls, question):
        question = str(question).lower().strip()

        if not question:
            return "reject"

        # DAX function names themselves are domain signals. This prevents
        # questions such as "SUM vs SUMX" from being rejected merely because
        # the user did not type the word DAX or Power BI.
        if cls.is_dax_function_question(question):
            return "web"

        if cls.is_off_topic(question) and not cls.is_power_bi_domain(question):
            return "reject"

        if cls.is_power_bi_domain(question):
            if cls.requires_web(question) or cls.needs_web_for_knowledge_gap(question):
                return "web"
            if cls.is_local_core(question):
                return "local"
            return "local"

        implicit_bi_patterns = [
            r"\bsemantic\b", r"\bdata\s+model\b", r"\bdata\s+modeling\b",
            r"\bdata\s+visualization\b", r"\bbusiness\s+analytics\b",
            r"\bbusiness\s+intelligence\b", r"\bdax\b", r"\bmeasure(s)?\b",
            r"\bquery\s+folding\b", r"\bincremental\s+refresh\b", r"\bdirectquery\b",
            r"\bdata\s+refresh\b",
        ]
        if any(re.search(pattern, question) for pattern in implicit_bi_patterns):
            return "web" if cls.requires_web(question) or cls.needs_web_for_knowledge_gap(question) else "local"

        return "reject"

    @staticmethod
    def rejection_message():
        return (
            "I'm a Power BI-focused assistant. "
            "I can help with Power BI, Microsoft Fabric, Power Query, DAX, semantic models, "
            "Power BI Service, Power BI Desktop, business intelligence, analytics, "
            "performance, administration, and related BI technologies."
        )
