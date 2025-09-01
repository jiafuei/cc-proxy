Here's a comprehensive JSON example for a `GenerateContentRequest` and its corresponding `GenerateContentResponse` using the Gemini API, with a description of each field based on the provided context.

---

### Comprehensive `GenerateContentRequest` JSON Example

This example includes various optional fields to demonstrate a wide range of capabilities.

```json
{
  "model": "models/gemini-2.0-flash",
  "contents": [
    {
      "role": "user",
      "parts": [
        {
          "text": "Tell me a story about a magic backpack."
        },
        {
          "file_data": {
            "mime_type": "image/jpeg",
            "file_uri": "https://example.com/image.jpg"
          }
        }
      ]
    }
  ],
  "tools": [
    {
      "function_declarations": [
        {
          "name": "get_weather",
          "description": "Get the current weather in a given location.",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA"
              }
            },
            "required": ["location"]
          }
        }
      ]
    }
  ],
  "toolConfig": {
    "functionCallingConfig": {
      "mode": "ANY"
    }
  },
  "safetySettings": [
    {
      "category": "HARM_CATEGORY_HATE_SPEECH",
      "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
      "category": "HARM_CATEGORY_HARASSMENT",
      "threshold": "BLOCK_ONLY_HIGH"
    }
  ],
  "systemInstruction": {
    "parts": [
      {
        "text": "You are a helpful assistant."
      }
    ]
  },
  "generationConfig": {
    "stopSequences": ["END_STORY"],
    "responseMimeType": "text/plain",
    "candidateCount": 1,
    "maxOutputTokens": 500,
    "temperature": 0.7,
    "topP": 0.9,
    "topK": 40,
    "seed": 12345,
    "presencePenalty": 0.1,
    "frequencyPenalty": 0.1,
    "responseLogprobs": true,
    "logprobs": 3,
    "enableEnhancedCivicAnswers": false,
    "speechConfig": {
      "voiceConfig": {
        "prebuiltVoiceConfig": {
          "voiceName": "en-US-Standard-A"
        }
      },
      "languageCode": "en-US"
    },
    "thinkingConfig": {
      "includeThoughts": true,
      "thinkingBudget": 100
    },
    "mediaResolution": "MEDIA_RESOLUTION_HIGH"
  },
  "cachedContent": "cachedContents/my-cached-document"
}
```

---

### Description of `GenerateContentRequest` Fields

*   **`model`** (`string`, Required): The name of the `Model` to use for generating the completion. Format: `models/{model}`.
*   **`contents[]`** (`object (Content)`, Required): The content of the current conversation with the model. For single-turn queries, this is a single instance. For multi-turn queries like chat, this is a repeated field that contains the conversation history and the latest request.
    *   **`role`** (`string`): The role of the author of this content. Can be "user" or "model".
    *   **`parts[]`** (`object (Part)`): A list of parts that make up the content.
        *   **`text`** (`string`): Plain text content.
        *   **`file_data`** (`object`): Data for a file, including its MIME type and URI.
            *   **`mime_type`** (`string`): The MIME type of the file.
            *   **`file_uri`** (`string`): The URI of the file.
*   **`tools[]`** (`object (Tool)`, Optional): A list of `Tools` the `Model` may use to generate the next response. A `Tool` is a piece of code that enables the system to interact with external systems to perform an action, or set of actions, outside of knowledge and scope of the `Model`. Supported `Tool`s are `Function` and `codeExecution`.
    *   **`function_declarations[]`** (`object (FunctionDeclaration)`): Describes a function that the model can call.
        *   **`name`** (`string`): The name of the function.
        *   **`description`** (`string`): A description of what the function does.
        *   **`parameters`** (`object (Schema)`): The parameters the function accepts, described as an OpenAPI schema.
*   **`toolConfig`** (`object (ToolConfig)`, Optional): Tool configuration for any `Tool` specified in the request.
    *   **`functionCallingConfig`** (`object (FunctionCallingConfig)`): Configuration for how the model should use functions.
        *   **`mode`** (`enum`): The mode for function calling (e.g., "ANY" to allow the model to call any function).
*   **`safetySettings[]`** (`object (SafetySetting)`, Optional): A list of unique `SafetySetting` instances for blocking unsafe content. This will be enforced on the `GenerateContentRequest.contents` and `GenerateContentResponse.candidates`. There should not be more than one setting for each `SafetyCategory` type.
    *   **`category`** (`enum (HarmCategory)`, Required): The category for this setting (e.g., "HARM_CATEGORY_HATE_SPEECH").
    *   **`threshold`** (`enum (HarmBlockThreshold)`, Required): Controls the probability threshold at which harm is blocked (e.g., "BLOCK_MEDIUM_AND_ABOVE").
*   **`systemInstruction`** (`object (Content)`, Optional): Developer set system instruction(s). Currently, text only.
    *   **`parts[]`** (`object (Part)`): A list of parts that make up the system instruction.
        *   **`text`** (`string`): Plain text content of the instruction.
*   **`generationConfig`** (`object (GenerationConfig)`, Optional): Configuration options for model generation and outputs.
    *   **`stopSequences[]`** (`string`, Optional): The set of character sequences (up to 5) that will stop output generation.
    *   **`responseMimeType`** (`string`, Optional): MIME type of the generated candidate text (e.g., "text/plain", "application/json").
    *   **`candidateCount`** (`integer`, Optional): Number of generated responses to return. Defaults to 1.
    *   **`maxOutputTokens`** (`integer`, Optional): The maximum number of tokens to include in a response candidate.
    *   **`temperature`** (`number`, Optional): Controls the randomness of the output. Values can range from [0.0, 2.0].
    *   **`topP`** (`number`, Optional): The maximum cumulative probability of tokens to consider when sampling.
    *   **`topK`** (`integer`, Optional): The maximum number of tokens to consider when sampling.
    *   **`seed`** (`integer`, Optional): Seed used in decoding. If not set, the request uses a randomly generated seed.
    *   **`presencePenalty`** (`number`, Optional): Presence penalty applied to the next token's logprobs if the token has already been seen in the response.
    *   **`frequencyPenalty`** (`number`, Optional): Frequency penalty applied to the next token's logprobs, multiplied by the number of times each token has been seen in the response so far.
    *   **`responseLogprobs`** (`boolean`, Optional): If true, export the logprobs results in response.
    *   **`logprobs`** (`integer`, Optional): Only valid if `responseLogprobs=True`. Sets the number of top logprobs to return at each decoding step. Range [1, 5].
    *   **`enableEnhancedCivicAnswers`** (`boolean`, Optional): Enables enhanced civic answers.
    *   **`speechConfig`** (`object (SpeechConfig)`, Optional): The speech generation config.
        *   **`voiceConfig`** (`object (VoiceConfig)`): The configuration in case of single-voice output.
            *   **`prebuiltVoiceConfig`** (`object (PrebuiltVoiceConfig)`): The configuration for the prebuilt voice to use.
                *   **`voiceName`** (`string`): The name of the preset voice to use.
        *   **`languageCode`** (`string`, Optional): Language code (in BCP 47 format, e.g. "en-US") for speech synthesis.
    *   **`thinkingConfig`** (`object (ThinkingConfig)`, Optional): Config for thinking features.
        *   **`includeThoughts`** (`boolean`): Indicates whether to include thoughts in the response.
        *   **`thinkingBudget`** (`integer`): The number of thoughts tokens that the model should generate.
    *   **`mediaResolution`** (`enum (MediaResolution)`, Optional): If specified, the media resolution specified will be used (e.g., "MEDIA_RESOLUTION_HIGH").
*   **`cachedContent`** (`string`, Optional): The name of the content cached to use as context to serve the prediction. Format: `cachedContents/{cachedContent}`.

---

### Comprehensive `GenerateContentResponse` JSON Example

This example shows a response with various metadata and potential grounding information.

```json
{
  "candidates": [
    {
      "content": {
        "role": "model",
        "parts": [
          {
            "text": "Once upon a time, in a bustling city, lived a young adventurer named Alex. Alex possessed a magical backpack that could conjure anything they wished for. One day, Alex wished for a map to a hidden treasure, and instantly, a shimmering map appeared. END_STORY"
          }
        ]
      },
      "finishReason": "STOP",
      "safetyRatings": [
        {
          "category": "HARM_CATEGORY_HATE_SPEECH",
          "probability": "NEGLIGIBLE",
          "blocked": false
        },
        {
          "category": "HARM_CATEGORY_HARASSMENT",
          "probability": "NEGLIGIBLE",
          "blocked": false
        }
      ],
      "citationMetadata": {
        "citationSources": [
          {
            "startIndex": 0,
            "endIndex": 10,
            "uri": "https://example.com/source1.html",
            "license": "CC BY 4.0"
          }
        ]
      },
      "tokenCount": 45,
      "groundingAttributions": [
        {
          "sourceId": {
            "groundingPassage": {
              "passageId": "passage-123",
              "partIndex": 0
            }
          },
          "content": {
            "parts": [
              {
                "text": "Alex's backpack was known for its magical properties."
              }
            ]
          }
        }
      ],
      "groundingMetadata": {
        "groundingChunks": [
          {
            "web": {
              "uri": "https://example.com/web-chunk-1.html",
              "title": "Magic Backpack Lore"
            }
          }
        ],
        "groundingSupports": [
          {
            "groundingChunkIndices": [0],
            "confidenceScores": [0.95],
            "segment": {
              "partIndex": 0,
              "startIndex": 15,
              "endIndex": 25,
              "text": "magic backpack"
            }
          }
        ],
        "webSearchQueries": ["magic backpack stories"],
        "searchEntryPoint": {
          "renderedContent": "Search results for 'magic backpack stories'",
          "sdkBlob": "base64encodedjson"
        },
        "retrievalMetadata": {
          "googleSearchDynamicRetrievalScore": 0.85
        }
      },
      "avgLogprobs": -0.123,
      "logprobsResult": {
        "topCandidates": [
          {
            "candidates": [
              {
                "token": "Once",
                "tokenId": 101,
                "logProbability": -0.01
              },
              {
                "token": "A",
                "tokenId": 102,
                "logProbability": -0.05
              }
            ]
          }
        ],
        "chosenCandidates": [
          {
            "token": "Once",
            "tokenId": 101,
            "logProbability": -0.01
          }
        ]
      },
      "urlContextMetadata": {
        "urlMetadata": [
          {
            "retrievedUrl": "https://example.com/retrieved-url.html",
            "urlRetrievalStatus": "URL_RETRIEVAL_STATUS_SUCCESS"
          }
        ]
      },
      "index": 0
    }
  ],
  "promptFeedback": {
    "blockReason": "SAFETY",
    "safetyRatings": [
      {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "probability": "HIGH",
        "blocked": true
      }
    ]
  },
  "usageMetadata": {
    "promptTokenCount": 50,
    "cachedContentTokenCount": 20,
    "candidatesTokenCount": 45,
    "toolUsePromptTokenCount": 5,
    "thoughtsTokenCount": 10,
    "totalTokenCount": 105,
    "promptTokensDetails": [
      {
        "modality": "TEXT",
        "tokenCount": 30
      },
      {
        "modality": "IMAGE",
        "tokenCount": 20
      }
    ],
    "cacheTokensDetails": [
      {
        "modality": "TEXT",
        "tokenCount": 20
      }
    ],
    "candidatesTokensDetails": [
      {
        "modality": "TEXT",
        "tokenCount": 45
      }
    ],
    "toolUsePromptTokensDetails": [
      {
        "modality": "TEXT",
        "tokenCount": 5
      }
    ]
  },
  "modelVersion": "gemini-2.0-flash",
  "responseId": "some-unique-response-id"
}
```

---

### Description of `GenerateContentResponse` Fields

*   **`candidates[]`** (`object (Candidate)`): Candidate responses from the model.
    *   **`content`** (`object (Content)`): Generated content returned from the model.
        *   **`role`** (`string`): The role of the author of this content.
        *   **`parts[]`** (`object (Part)`): A list of parts that make up the content.
            *   **`text`** (`string`): Plain text content.
    *   **`finishReason`** (`enum (FinishReason)`, Optional): The reason why the model stopped generating tokens (e.g., "STOP", "SAFETY", "MAX_TOKENS").
    *   **`safetyRatings[]`** (`object (SafetyRating)`): List of ratings for the safety of a response candidate.
        *   **`category`** (`enum (HarmCategory)`, Required): The category for this rating.
        *   **`probability`** (`enum (HarmProbability)`, Required): The probability of harm for this content (e.g., "NEGLIGIBLE", "HIGH").
        *   **`blocked`** (`boolean`): Was this content blocked because of this rating?
    *   **`citationMetadata`** (`object (CitationMetadata)`): Citation information for model-generated candidate.
        *   **`citationSources[]`** (`object (CitationSource)`): Citations to sources for a specific response.
            *   **`startIndex`** (`integer`, Optional): Start of segment of the response that is attributed to this source, measured in bytes.
            *   **`endIndex`** (`integer`, Optional): End of the attributed segment, exclusive.
            *   **`uri`** (`string`, Optional): URI that is attributed as a source.
            *   **`license`** (`string`, Optional): License for the GitHub project (required for code citations).
    *   **`tokenCount`** (`integer`): Token count for this candidate.
    *   **`groundingAttributions[]`** (`object (GroundingAttribution)`): Attribution information for sources that contributed to a grounded answer.
        *   **`sourceId`** (`object (AttributionSourceId)`): Identifier for the source contributing to this attribution.
            *   **`groundingPassage`** (`object (GroundingPassageId)`): Identifier for an inline passage.
                *   **`passageId`** (`string`): ID of the passage.
                *   **`partIndex`** (`integer`): Index of the part within the passage content.
            *   **`semanticRetrieverChunk`** (`object (SemanticRetrieverChunk)`): Identifier for a `Chunk` fetched via Semantic Retriever.
                *   **`source`** (`string`): Name of the source (e.g., `corpora/123`).
                *   **`chunk`** (`string`): Name of the `Chunk` (e.g., `corpora/123/documents/abc/chunks/xyz`).
        *   **`content`** (`object (Content)`): Grounding source content that makes up this attribution.
    *   **`groundingMetadata`** (`object (GroundingMetadata)`): Grounding metadata for the candidate.
        *   **`groundingChunks[]`** (`object (GroundingChunk)`): List of supporting references retrieved from specified grounding source.
            *   **`web`** (`object (Web)`): Grounding chunk from the web.
                *   **`uri`** (`string`): URI reference of the chunk.
                *   **`title`** (`string`): Title of the chunk.
        *   **`groundingSupports[]`** (`object (GroundingSupport)`): List of grounding support.
            *   **`groundingChunkIndices[]`** (`integer`): A list of indices (into 'grounding_chunk') specifying the citations.
            *   **`confidenceScores[]`** (`number`): Confidence score of the support references. Ranges from 0 to 1.
            *   **`segment`** (`object (Segment)`): Segment of the content this support belongs to.
                *   **`partIndex`** (`integer`): The index of a Part object within its parent Content object.
                *   **`startIndex`** (`integer`): Start index in the given Part, measured in bytes.
                *   **`endIndex`** (`integer`): End index in the given Part, measured in bytes.
                *   **`text`** (`string`): The text corresponding to the segment.
        *   **`webSearchQueries[]`** (`string`): Web search queries for the following-up web search.
        *   **`searchEntryPoint`** (`object (SearchEntryPoint)`, Optional): Google search entry for the following-up web searches.
            *   **`renderedContent`** (`string`, Optional): Web content snippet.
            *   **`sdkBlob`** (`string (bytes format)`, Optional): Base64 encoded JSON representing array of  tuple.
        *   **`retrievalMetadata`** (`object (RetrievalMetadata)`): Metadata related to retrieval in the grounding flow.
            *   **`googleSearchDynamicRetrievalScore`** (`number`, Optional): Score indicating how likely information from google search could help answer the prompt.
    *   **`avgLogprobs`** (`number`): Average log probability score of the candidate.
    *   **`logprobsResult`** (`object (LogprobsResult)`): Log-likelihood scores for the response tokens and top tokens.
        *   **`topCandidates[]`** (`object (TopCandidates)`): Candidates with top log probabilities at each decoding step.
            *   **`candidates[]`** (`object (Candidate)`): Sorted by log probability in descending order.
                *   **`token`** (`string`): The candidate’s token string value.
                *   **`tokenId`** (`integer`): The candidate’s token id value.
                *   **`logProbability`** (`number`): The candidate's log probability.
        *   **`chosenCandidates[]`** (`object (Candidate)`): The chosen candidates (tokens) at each decoding step.
    *   **`urlContextMetadata`** (`object (UrlContextMetadata)`): Metadata related to url context retrieval tool.
        *   **`urlMetadata[]`** (`object (UrlMetadata)`): List of url context.
            *   **`retrievedUrl`** (`string`): Retrieved url by the tool.
            *   **`urlRetrievalStatus`** (`enum (UrlRetrievalStatus)`): Status of the url retrieval (e.g., "URL_RETRIEVAL_STATUS_SUCCESS").
    *   **`index`** (`integer`): Index of the candidate in the list of response candidates.
*   **`promptFeedback`** (`object (PromptFeedback)`): Returns the prompt's feedback related to the content filters.
    *   **`blockReason`** (`enum (BlockReason)`, Optional): If set, the prompt was blocked and no candidates are returned (e.g., "SAFETY", "BLOCKLIST").
    *   **`safetyRatings[]`** (`object (SafetyRating)`): Ratings for safety of the prompt.
*   **`usageMetadata`** (`object (UsageMetadata)`): Metadata on the generation requests' token usage.
    *   **`promptTokenCount`** (`integer`): Number of tokens in the prompt.
    *   **`cachedContentTokenCount`** (`integer`): Number of tokens in the cached part of the prompt.
    *   **`candidatesTokenCount`** (`integer`): Total number of tokens across all the generated response candidates.
    *   **`toolUsePromptTokenCount`** (`integer`): Number of tokens present in tool-use prompt(s).
    *   **`thoughtsTokenCount`** (`integer`): Number of tokens of thoughts for thinking models.
    *   **`totalTokenCount`** (`integer`): Total token count for the generation request (prompt + response candidates).
    *   **`promptTokensDetails[]`** (`object (ModalityTokenCount)`): List of modalities that were processed in the request input.
        *   **`modality`** (`enum (Modality)`): The modality associated with this token count (e.g., "TEXT", "IMAGE").
        *   **`tokenCount`** (`integer`): Number of tokens for this modality.
    *   **`cacheTokensDetails[]`** (`object (ModalityTokenCount)`): List of modalities of the cached content in the request input.
    *   **`candidatesTokensDetails[]`** (`object (ModalityTokenCount)`): List of modalities that were returned in the response.
    *   **`toolUsePromptTokensDetails[]`** (`object (ModalityTokenCount)`): List of modalities that were processed for tool-use request inputs.
*   **`modelVersion`** (`string`): The model version used to generate the response.
*   **`responseId`** (`string`): responseId is used to identify each response.