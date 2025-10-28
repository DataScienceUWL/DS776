# Instructor Note: Add LangChain to Lesson 11

## Background
During development of Lesson 7, we discussed that LangChain provides similar wrapper functions to our `llm_generate()` but with more advanced features and production-oriented capabilities.

## Action Item
**Add LangChain coverage to Lesson 11 (Text Generation)**

### Why Lesson 11?
- Text generation is where LangChain shines most
- Students will have already mastered the basics with `llm_generate()`
- Good time to introduce more advanced, production-ready tools
- LangChain's chains and memory features are particularly relevant for text generation tasks

### Suggested Content

1. **Introduction to LangChain**
   - Brief overview of what LangChain is and why it exists
   - Comparison with our `llm_generate()` approach
   - When to use each (simple tasks vs. production applications)

2. **LangChain Demonstration**
   - Basic LLM wrapper usage (`ChatOpenAI`, `ChatAnthropic`, `ChatGoogleGenerativeAI`)
   - Prompt templates (`PromptTemplate`, `ChatPromptTemplate`)
   - Structured output and JSON schemas
   - Chains for combining operations
   - Memory for conversation history

3. **Practical Example**
   - Build a simple chatbot or text generation workflow using LangChain
   - Compare it side-by-side with equivalent `llm_generate()` code
   - Highlight when the extra complexity is worth it

### Installation Note
Will need to add LangChain dependencies to `introdl` package or have students install separately:
```bash
pip install langchain langchain-openai langchain-anthropic langchain-google-genai
```

### References
- LangChain Docs: https://python.langchain.com/docs/get_started/introduction
- LangChain Expression Language (LCEL): https://python.langchain.com/docs/expression_language/

## Related Tools to Mention (Optional)
- **LlamaIndex**: Similar framework, more focused on RAG (Retrieval-Augmented Generation)
- **Haystack**: Another LLM application framework
- **Guidance**: Microsoft's framework for controlling generation

## Status
- [ ] Add LangChain installation to course package or lesson requirements
- [ ] Create LangChain introduction section in L11_1_Text_Generation.ipynb
- [ ] Develop practical demonstration example
- [ ] Test all code examples
- [ ] Update lesson overview to mention LangChain coverage
