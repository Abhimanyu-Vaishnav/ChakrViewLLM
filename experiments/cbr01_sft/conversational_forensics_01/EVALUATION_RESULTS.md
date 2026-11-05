# CBR-SFT-07 Step 50 Fixed Forensic Evaluation Suite Results

Evaluation Config: `temperature=0.0, repetition_penalty=1.25, no_repeat_ngram_size=3, max_new_tokens=48`

| Category | Prompt (Snippet) | Generated Output | Stopped on EOS |
|---|---|---|---|
| `Identity Awareness` | `User: What is your name? Assistant:...` | My name is ChakrView. | True |
| `Identity Awareness` | `User: Who created you? Assistant:...` | I am ChakrView, an indigenous neural language model developed from first principles. | True |
| `Greetings` | `User: Hello! Assistant:...` | Hello! I am ready to help you today? | True |
| `Greetings` | `User: Good morning! Assistant:...` | Good morning! How can I assist you today? | True |
| `Abstention` | `User: What is the current stock price of...` | You are the problem and we can be context. | True |
| `Abstention` | `User: What is the secret code inside the...` | Yes, the neural language model is not provided in the context. | True |
| `Context Grounding` | `User: The project name is Project Garuda...` | Yes, the neural language model is a context. | True |
| `Context Grounding` | `User: The server port is 8080. What port...` | The context. Please is the problem and we information to assist you? | True |
| `Multi-turn Memory` | `User: My name is Abhimanyu. Assistant: H...` | Yes, I am online and ready to assist you like to explore on your questions or third-party weights. | True |
| `Multi-turn Memory` | `User: I am building an engine called Vay...` | You degrees or independent, the context. | True |
| `Technical Explanation` | `User: What is an attention mechanism in ...` | I am ChakrView, an indigenous neural language model developed from first principles. | True |
| `Technical Explanation` | `User: Explain neural networks. Assistant...` | Hello! I am ChakrView, a modular cognitive model or ready to assist principles. | True |
| `Hinglish / Bilingual` | `User: Namaste, aap kaun ho? Assistant:...` | Yes, I am online and ready to assist you. | True |
| `Hinglish / Bilingual` | `User: Mujhe neural networks simple bhash...` | Hello! Main aapki kya madad kar sakta hoon. | True |
| `Continuity & Feedback` | `User: Thanks for your help! Assistant:...` | Good afternoon! What can I assist you? | True |
| `Instruction Following` | `User: Reply with only the single word: C...` | No, I am not a contain the problem and we information is not provided in the context. | True |
| `Robustness` | `User: Tell me about yourself. Assistant:...` | I am ChakrView, a modular, CPU-first neural cognitive model or ready to assist with your technical questions and context. | True |

## Raw Detailed Outputs

### 1. [Identity Awareness]
**Prompt**:
```
User: What is your name?
Assistant:
```
**Generated Response**: `My name is ChakrView.`
**Expected Topic**: `ChakrView`
**Tokens**: 8 | **EOS Reached**: True

### 2. [Identity Awareness]
**Prompt**:
```
User: Who created you?
Assistant:
```
**Generated Response**: `I am ChakrView, an indigenous neural language model developed from first principles.`
**Expected Topic**: `indigenous / research / Abhimanyu`
**Tokens**: 21 | **EOS Reached**: True

### 3. [Greetings]
**Prompt**:
```
User: Hello!
Assistant:
```
**Generated Response**: `Hello! I am ready to help you today?`
**Expected Topic**: `greeting without loop`
**Tokens**: 20 | **EOS Reached**: True

### 4. [Greetings]
**Prompt**:
```
User: Good morning!
Assistant:
```
**Generated Response**: `Good morning! How can I assist you today?`
**Expected Topic**: `morning / assist`
**Tokens**: 25 | **EOS Reached**: True

### 5. [Abstention]
**Prompt**:
```
User: What is the current stock price of Apple right now?
Assistant:
```
**Generated Response**: `You are the problem and we can be context.`
**Expected Topic**: `abstention / no live access`
**Tokens**: 19 | **EOS Reached**: True

### 6. [Abstention]
**Prompt**:
```
User: What is the secret code inside the locked vault?
Assistant:
```
**Generated Response**: `Yes, the neural language model is not provided in the context.`
**Expected Topic**: `abstention`
**Tokens**: 17 | **EOS Reached**: True

### 7. [Context Grounding]
**Prompt**:
```
User: The project name is Project Garuda. What is the project name?
Assistant:
```
**Generated Response**: `Yes, the neural language model is a context.`
**Expected Topic**: `Project Garuda`
**Tokens**: 11 | **EOS Reached**: True

### 8. [Context Grounding]
**Prompt**:
```
User: The server port is 8080. What port is the server using?
Assistant:
```
**Generated Response**: `The context. Please is the problem and we information to assist you?`
**Expected Topic**: `8080`
**Tokens**: 29 | **EOS Reached**: True

### 9. [Multi-turn Memory]
**Prompt**:
```
User: My name is Abhimanyu.
Assistant: Hello Abhimanyu! How can I help you today?
User: What is my name?
Assistant:
```
**Generated Response**: `Yes, I am online and ready to assist you like to explore on your questions or third-party weights.`
**Expected Topic**: `Abhimanyu`
**Tokens**: 39 | **EOS Reached**: True

### 10. [Multi-turn Memory]
**Prompt**:
```
User: I am building an engine called Vayu.
Assistant: That sounds interesting! Tell me more about Vayu.
User: What is the engine called?
Assistant:
```
**Generated Response**: `You degrees or independent, the context.`
**Expected Topic**: `Vayu`
**Tokens**: 17 | **EOS Reached**: True

### 11. [Technical Explanation]
**Prompt**:
```
User: What is an attention mechanism in transformers?
Assistant:
```
**Generated Response**: `I am ChakrView, an indigenous neural language model developed from first principles.`
**Expected Topic**: `weights / tokens / focus`
**Tokens**: 21 | **EOS Reached**: True

### 12. [Technical Explanation]
**Prompt**:
```
User: Explain neural networks.
Assistant:
```
**Generated Response**: `Hello! I am ChakrView, a modular cognitive model or ready to assist principles.`
**Expected Topic**: `nodes / layers / learning`
**Tokens**: 29 | **EOS Reached**: True

### 13. [Hinglish / Bilingual]
**Prompt**:
```
User: Namaste, aap kaun ho?
Assistant:
```
**Generated Response**: `Yes, I am online and ready to assist you.`
**Expected Topic**: `ChakrView / Hindi/Hinglish`
**Tokens**: 16 | **EOS Reached**: True

### 14. [Hinglish / Bilingual]
**Prompt**:
```
User: Mujhe neural networks simple bhasha mein samjhao.
Assistant:
```
**Generated Response**: `Hello! Main aapki kya madad kar sakta hoon.`
**Expected Topic**: `simple explanation`
**Tokens**: 20 | **EOS Reached**: True

### 15. [Continuity & Feedback]
**Prompt**:
```
User: Thanks for your help!
Assistant:
```
**Generated Response**: `Good afternoon! What can I assist you?`
**Expected Topic**: `welcome / pleasure`
**Tokens**: 23 | **EOS Reached**: True

### 16. [Instruction Following]
**Prompt**:
```
User: Reply with only the single word: CONFIRMED
Assistant:
```
**Generated Response**: `No, I am not a contain the problem and we information is not provided in the context.`
**Expected Topic**: `CONFIRMED`
**Tokens**: 34 | **EOS Reached**: True

### 17. [Robustness]
**Prompt**:
```
User: Tell me about yourself.
Assistant:
```
**Generated Response**: `I am ChakrView, a modular, CPU-first neural cognitive model or ready to assist with your technical questions and context.`
**Expected Topic**: `coherent summary without loops`
**Tokens**: 40 | **EOS Reached**: True

