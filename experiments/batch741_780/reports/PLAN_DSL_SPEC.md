# Plan Domain-Specific Language (Plan DSL) Specification
## Batches 741–780: Autoregressive Plan DSL, Variable-Length Neural Planning & Safe Execution

**Specification Version:** 1.0.0  
**Status:** FROZEN  
**Contract Date:** 2026-10-15  
**Governing Module:** `chakrview/cognition/plan_dsl.py`  

---

### 1. Specification & Grammar Overview

The Plan DSL is a compact, token-efficient, line-oriented declarative language for emitting variable-length task plans, subtasks, dependencies, dynamic intermediate bindings, and explicit termination.

#### Formal EBNF Grammar:

```ebnf
Plan ::= Header SubtaskList [Clarification] EndOfPlan

Header ::= "GOAL:" GoalType "\n"
GoalType ::= "compound_arithmetic" | "text_transform_chain" | "evidence_reasoning" | "clarification_needed" | "unsupported"

SubtaskList ::= Subtask+
Subtask ::= "STEP" StepIndex ":" ActionType "|" ToolName "|" ArgList "|" DepList "\n"

StepIndex ::= [0-9]+
ActionType ::= "tool_call" | "text_transform" | "clarification" | "reasoning"
ToolName ::= "calculator" | "text_utility" | "none" | String
ArgList ::= "args={" KeyVal ("," KeyVal)* "}" | "args={}"
KeyVal ::= Key "=" Value
DepList ::= "deps=[" [StepIndex ("," StepIndex)*] "]"

Clarification ::= "CLARIFICATION: NEEDED" "\n" | "CLARIFICATION: NONE" "\n"
EndOfPlan ::= "<END_PLAN>"
```

---

### 2. Concrete Examples

#### 2.1 Three-Hop Arithmetic Chain
```
GOAL: compound_arithmetic
STEP 0: tool_call | calculator | args={expr=12*4} | deps=[]
STEP 1: tool_call | calculator | args={expr=$0/2} | deps=[0]
STEP 2: tool_call | calculator | args={expr=$1+10} | deps=[1]
CLARIFICATION: NONE
<END_PLAN>
```

#### 2.2 Text Drafting and Two-Step Transformation
```
GOAL: text_transform_chain
STEP 0: text_transform | text_utility | args={text='  ChakrView Core  ',op=strip} | deps=[]
STEP 1: text_transform | text_utility | args={text=$0,op=lowercase} | deps=[0]
CLARIFICATION: NONE
<END_PLAN>
```

#### 2.3 Targeted Clarification on Missing Information
```
GOAL: clarification_needed
STEP 0: clarification | none | args={missing=discount_rate} | deps=[]
CLARIFICATION: NEEDED
<END_PLAN>
```

#### 2.4 Unsupported Tool Request
```
GOAL: unsupported
STEP 0: clarification | none | args={reason='unauthorized tool root_shell'} | deps=[]
CLARIFICATION: NEEDED
<END_PLAN>
```

---

### 3. Parsing & Validation Rules

1. **Deterministic Parsing (`PlanDSLParser`):**
   - Strips leading/trailing whitespace.
   - Extracts goal, sequential steps, typed tools, argument dictionaries, and dependencies.
   - Normalizes variable bindings (`$0` -> output of Step 0).
2. **DAG Validation (`PlanDSLValidator`):**
   - **Acyclicity:** Every dependency $dep \in deps$ must satisfy $dep < current\_step$.
   - **Tool Authorization:** Only registered runtime tools are permitted. Unknown tool names flag the subtask as unauthorized.
   - **Termination:** Plans missing `<END_PLAN>` are marked truncated and rejected unless partial execution is permitted.
   - **No Gold Fabrication:** The parser and validator strictly evaluate model output. They **never** substitute gold steps or synthesize missing arguments.
