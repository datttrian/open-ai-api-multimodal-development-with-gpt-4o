# OpenAI API: Multimodal Development with GPT-4o

## Adding Functionality through Additional Function Calls

One of the powerful features of function calling in the OpenAI API is
the ability to add multiple different function call options to be
executed independently or in sequence. You can do this by adding
additional objects to the `function` object in the `tools` array for the
`create` request.

The exercise files for this course include  a file named
`utils/function_call.py` with a built out call to the `itemize_receipt`
function. This function call kicks in any time GPT-4o detects an image
of a receipt, and outputs the data from that receipt.

The relevant parts of the code look like this:

```python
function_call = [
  {
    "type": "function",
    "function": {
      "name": "itemize_receipt",
      "description": "Itemize a receipt from an image",
      "parameters": {
        # Description of properties
      },
      "required": ["vendor","date","items","payment_method"],
    }
  }
]
```

### Adding New Functionality

Let's say you want to turn the example project into an app, and give the
user the ability to not only add receipts to the spreadsheet, but also
ask questions about the data in the spreadsheet.

(Note: This requires building a UI with both image and text input
fields, which falls outside the scope of this course, so in the
following example we'll focus on the API interaction and function
calling.)

To add this functionality, you first need to build the actual code to
process the spreadsheet data. In Python you can set that up as a new
module in the `utils/` folder called `custom_sum.py` like this:

```python
import pandas as pd

def custom_sum(time_range, expenses_df):
    """
    Function to calculate the sum of prices based on a specified time range.

    Parameters:
    time_range (dict): Dictionary specifying the time range.
                 Possible keys: 'specific_date', 'start_date', 'end_date', 'month', 'year'
    expenses_df (pd.DataFrame): DataFrame containing the data.

    Returns:
    float: Sum of the prices within the specified time range.
    """

    # Convert the 'Date' column from string to datetime format
    expenses_df['Date'] = pd.to_datetime(expenses_df['Date'])

    # Filter the DataFrame based on the specified time range
    if 'specific_date' in time_range:
        specific_date = pd.to_datetime(time_range['specific_date'])
        filtered_df = expenses_df[expenses_df['Date'] == specific_date]

    elif 'start_date' in time_range and 'end_date' in time_range:
        start_date = pd.to_datetime(time_range['start_date'])
        end_date = pd.to_datetime(time_range['end_date'])
        filtered_df = expenses_df[(expenses_df['Date'] >= start_date) & (expenses_df['Date'] <= end_date)]

    elif 'month' in time_range and 'year' in time_range:
        month = time_range['month']
        year = time_range['year']
        filtered_df = expenses_df[(expenses_df['Date'].dt.month == month) & (expenses_df['Date'].dt.year == year)]

    elif 'year' in time_range:
        year = time_range['year']
        filtered_df = expenses_df[expenses_df['Date'].dt.year == year]

    else:
        raise ValueError("Invalid time specification.")

    # Calculate the sum
    return filtered_df['Price'].sum()
```

To use this function, you need to pass in two parameters:

-   `time_range`: a dictionary specifying the time range you want to
    calculate the sum for.
-   `expenses_df`: the expenses DataFrame containing the data.

The `time_range` dictionary can have the following formats:

```python
# A specific date
time_range = {'specific_date': '2024-05-11'}

# A range of dates
time_range = {'start_date': '2024-05-01', 'end_date': '2024-05-31'}

# A specific month and year
time_range = {'month': 5, 'year': 2024}

# A specific year
time_range = {'year': 2024}
```

### Building the New Function Call

Based on the possible variants of the `time_range` dictionary, you can
add a new function call object to the `function_call` array for the
`custom_sum` function like this:

```python
function_call = [
  {
    "type": "function",
    "function": {
      "name": "itemize_receipt",
      "description": "Itemize a receipt from an image",
      "parameters": {
        # Description of properties
      },
      "required": ["vendor","date","items","payment_method"],
    }
  },
  {
    "type": "function",
    "function": {
      "name": "custom_sum",
      "description": "Sum of expenses based on custom time period.",
      "parameters": {
        "type": "object",
        "properties": {
          "specific_date": {
            "type": "string",
            "format": "date",
            "description": "Specific date.",
          },
          "start_date": {
            "type": "string",
            "format": "date",
            "description": "If a specific date range, the start date.",
          },
          "end_date": {
            "type": "string",
            "format": "date",
            "description": "If a specific date range, the end date.",
          },
          "month": {
            "type": "integer",
            "description": "Month of the year. Must be combined with year.",
          },
          "year": {
            "type": "integer",
            "description": "Year. Can be stand-alone or combined with month.",
          },
        }
      }
    }
  }
]
```

A few things to note here:

-   The new function has the name `custom_sum` and a description
    instructing GPT-4o on when to trigger this function.
-   The `parameters` object contains the different possible keys for the
    `time_range` dictionary, each with a description of what it
    represents.
-   Some of the parameters have specific relationships, for example
    `start_date` and `end_date` must be used together to specify a date
    range, while `month` must be used together with `year` to specify a
    month in a specific year.
-   The `required` array is omitted from the `time_range` function
    object because the output parameters will depend on the user
    request.

### Basic Implementation

To test this new functionality with the existing code examples:

1.  Create a new `utils/custom_sum.py` file and add the code above.
2.  Add the new function call object to the `function_call` array in the
    `utils/function_call.py` file.
3.  Add a new `utils/get_sum.py` file and add the following code:

```python
import json
from utils.openai_init import openai_init
from utils.function_call import function_call
from utils.custom_sum import custom_sum

client = openai_init()

def get_sum(message, expenses_df):
  response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
      {"role": "system", "content": "If the image is a receipt, process the data. If it's not a receipt, ask for a receipt."},
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": message
          }
        ]
      },
    ],
    tools=function_call, # <-- Add the function_call schema from above
    tool_choice="auto",
    temperature=0.0,
  )

  print(f"time_range: {response.choices[0].message.tool_calls[0].function.arguments}")

  if response:
      # If the image is a receipt, process the data
      if (response.choices[0].message.tool_calls is not None and
          len(response.choices[0].message.tool_calls) > 0 and
          response.choices[0].message.tool_calls[0].function.name == "custom_sum"):

          # Extract receipt data from the response
          time_range = json.loads(response.choices[0].message.tool_calls[0].function.arguments)

          # Get the sum for the requested period
          requested_sum = custom_sum(time_range, expenses_df)

          return requested_sum

      else:
          print(f"\nWARNING:\n{response.choices[0].message.content}")
          return

  else:
      print("No response received from GPT-4o.")
      return
```

1.  In the \`modular-process.ipynb\` notebook, create a new code cell
    and add the following code:

```python
from utils.get_sum import get_sum

message = "How much did I spend between may 11 and 16th, 2024?"

print(f"Sum: {get_sum(message, expenses_df)}")
```

When you run this code, you'll get an output similar to this (if you use
the provided receipt data):

```python
time_range: {"start_date":"2024-05-11","end_date":"2024-05-16"}
Sum: 72.63
```

Now you can experiment with changing the message in the code block,
asking for a specific date, month, or year, and see how the `custom_sum`
function processes the data.

Chances are you'll discover some requests trip the system up, which is a
great opportunity to refine the system message and the function call
schema and a great reminder that the OpenAI API and the GPT models you
access through it are nondeterministic and can behave in unexpected ways
even with the most rigorous controls in place.

### Only the Beginning

This example carefully lifts the curtain on how powerful the OpenAI API
can be once you start thinking about it as more than just a chat
interface.  Combining multimodal prompting with function calling lets
you build advanced applications that can do many different things with
very little added code. The key is to lean on the AI for user input
interpretation, and on traditional tools like Python and Pandas for data
processing.