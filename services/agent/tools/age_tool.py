from datetime import datetime, date
from langchain.tools import BaseTool
import json


class AgeCalculatorTool(BaseTool):
    """
    Simple age tool for natural language DOBs.
    """
    name: str = "age_calculator"
    description: str = (
        "Use this tool when ever user wants to calculate the his/her age. "
        "The input to this tool should always be an JSON object but as a "
        "stringify object. This JSON object will have three keys named "
        "as 'month', 'day' and 'year'. If any of the key value is missing, "
        "pass the exact value as 'NAN'. Follow this isntructions for all "
        "the three keys. For example: If user has provided year and month "
        "but not day to calculate the age, input to the tool should be like "
        "{'month': 'month given by user', 'day': 'NAN', 'year': 'year given by"
        "user'}. Make sure that the JSON object is a string not a JSON "
        "object while passing."
    )
    return_direct: bool = False

    def _run(self, tool_input: str) -> str:
        print("before strip", tool_input)
        tool_input = tool_input.strip("'")
        print("after strip", tool_input)
        data_tool = json.loads(tool_input)
        if data_tool['year'].lower() == "nan":
            return "Your birth year is missing please provide that"
        elif data_tool['month'].lower() == "nan":
            return "Your birth month is missing please provide that"
        elif data_tool['day'].lower() == "nan":
            return "Your birth day is missing please provide that"
        dob_str = f"{data_tool['day']} {data_tool['month']} {data_tool['year']}"
        print(dob_str)

        birthdate = datetime.strptime(dob_str, "%d %m %Y").date()

        today = date.today()
        age = (
            today.year
            - birthdate.year
            - ((today.month, today.day) < (birthdate.month, birthdate.day))
        )
        return age
