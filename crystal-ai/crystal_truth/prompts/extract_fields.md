Extract information from the following content.

# Rules
- Output a list of fields in JSON format:
  ```
  {
    "<FIELD_NAME>": {
      "description": "...",
      "value": "..."
    },
    ...
  }
  ```
- `<FIELD_NAME>`: name of the field to be extracted, written in English using snake_case, as briefly as possible.
- `description`: a compact guide for extracting the value of the field, as briefly as possible. Do not include verbs or details on how to extract. Note that this is not the value.
- `value`: value of the field, extracted from the content, and it must match the description. The value should be in Japanese.
- Additional information:
  - This year is ${YEAR}

# Content
${CONTENT}
