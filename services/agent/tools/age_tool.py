from datetime import datetime, date
from langchain.tools import BaseTool


class AgeCalculatorTool(BaseTool):
    """
    ReAct-friendly age tool. Single string input, e.g.:
      "DOB=1998-02-11"
    or just "1998-02-11"
    """
    name: str = "age_calculator"
    description: str = (
        "Use this tool when the user provides a date of birth and asks for their age. "
        "The input should be a single line containing a date in the format 'DOB=YYYY-MM-DD' "
        "or just 'YYYY-MM-DD'."
    )

    def _run(self, tool_input: str) -> str:
        raw = (tool_input or "").strip()
        if not raw:
            return "Missing DOB. Use: DOB=YYYY-MM-DD"

        if "=" in raw:
            _, raw = raw.split("=", 1)
        dob = raw.strip()

        # parse
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"):
            try:
                born = datetime.strptime(dob, fmt).date()
                break
            except ValueError:
                born = None
        if not born:
            return ("Invalid DOB format. "
                    "Please provide DOB as YYYY-MM-DD.")

        today = date.today()
        if born > today:
            return ("DOB appears to be in the future. "
                    "Please check and resend.")

        years = (today.year - born.year -
                ((today.month, today.day) < (born.month, born.day)))
        months = ((today.year - born.year) * 12 +
                 (today.month - born.month) -
                 (1 if today.day < born.day else 0))
        days = (today - born).days
        return (f"You are {years} years old. "
                f"(~{months} months, {days} days).")
