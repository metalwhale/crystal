Extract information from the following content.

# Rules
- Output a list of fields in JSON format:
  ```json
  {
    "<FIELD_NAME_1>": "<VALUE_1>",
    "<FIELD_NAME_2>": "<VALUE_2>",
    ...
  }
  ```
- Values should be in Japanese
- Here is the list of fields to be extracted. Each field has a corresponding description. Please use these descriptions to extract the correct value for each field.
${FIELDS}

# Content
${CONTENT}
