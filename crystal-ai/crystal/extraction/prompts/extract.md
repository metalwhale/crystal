Extract information from the following content.

# Rules
- Output a list of fields in JSON format:
  ```json
  {
    "<FIELD_NAME_1>": {
      "value": "<VALUE_1>",
      "evidence": "<EVIDENCE_1>"
    },
    "<FIELD_NAME_2>": {
      "value": "<VALUE_2>",
      "evidence": "<EVIDENCE_2>"
    },
    ...
  }
  ```
- Values should be in Japanese.
- Evidence should be a snippet of content that explains the reason for the extraction and includes the corresponding value.
- Here is the list of fields to be extracted. Each field has a corresponding description. Please use these descriptions to extract the correct value for each field.
${FIELDS}

# Content
${CONTENT}
