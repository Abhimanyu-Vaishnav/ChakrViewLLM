# NR003: SFT Semantic & Distribution Audit

**Date:** 21 November 2026  
**Mission ID:** CV-GENESIS-NR-003  
**Project:** ChakrView  
**Dataset Inspected:** 1,476 unique instruction and dialogue pairs assembled in `train_sft.py`  

---

## 1. Composition & Target Imbalance Analysis

An exhaustive semantic triage of all 1,476 SFT pairs revealed:

- **Total Unique Instructions:** 1,476.
- **Top 25 Responses:** Account for **598 out of 1,476 examples (40.5% of the entire dataset)**.
- **Top Response Frequency Table:**
  1. `32x`: `'Acknowledged.'`
  2. `31x`: `'Green.'`
  3. `31x`: `'Yes.'`
  4. `31x`: `'System operational.'`
  5. `27x`: `'I am ChakrView, an indigenous conversational AI. I can assist with explanations, arithmetic, coding, and multilingual dialogue.'`
  6. `27x`: `'ChakrView is an indigenous neural architecture created by Abhimanyu Vaishnav.'`
  7. `27x`: `'जल जीवन का आधार है और स्वच्छ पेयजल के सीमित स्रोतों को भविष्य के लिए सुरक्षित रखना हमारा दायित्व है।...'`
  8. `26x`: `'किताबें पढ़ने से ज्ञान का विस्तार होता है, सोचने की क्षमता बढ़ती है और एकाग्रता में सुधार होता है।...'`
  9. `24x`: `'समय अत्यंत मूल्यवान संसाधन है; बीता हुआ समय कभी वापस नहीं आता, इसलिए इसका सदुपयोग करना चाहिए।...'`
  10. `24x`: `'Take a deep breath and take it one step at a time. I can help organize or break down any tasks you are facing.'`
  11. `24x`: `'That sounds exciting! Let me know if you need help with technical documentation, code snippets, or system architecture.'`
  12. `24x`: `'हिमालय पर्वतमाला भारत की उत्तरी सीमा पर स्थित एक विशाल प्राकृतिक सुरक्षा कवच है जो जलवायु को नियंत्रित करने में महत्वपूर्ण भूमिका निभाता है।...'`
  13. `24x`: `'इंटरनेट संचार, शिक्षा, व्यापार और सूचना के त्वरित आदान-प्रदान को सुगम बनाकर आधुनिक जीवनशैली का अभिन्न अंग बन चुका है।...'`
  14. `23x`: `'6'`
  15. `22x`: `'Good morning! I hope you have a productive day ahead. What are we working on?'`
  16. `22x`: `'I am ChakrView, an independent neural conversational assistant designed for efficient local execution.'`
  17. `68x`: Variations of `'You will have X remaining.'`

---

## 2. Artificial Prompt Duplication

In `ucip09s_balanced_train.jsonl`, single identical answers were paired with hundreds of trivially modified prompts appending synthetic session tags:
- `User: Hey there, what are your capabilities? (session 17)`
- `User: Hey there, what are your capabilities? (session 46)`
- `User: Hey there, what are your capabilities? (session 5)`
- `User: State the color of grass using only one word. (protocol check 27)`
- `User: State the color of grass using only one word. (protocol check 25)`

### Consequence:
The neural model learned to map arbitrary conversational tokens into high-probability attractor responses (such as `"You will have X remaining."`, `"Green."`, `"The ocean waves crash gently..."`, or `"Kindness costs nothing."`), rather than learning semantic associations between prompt content and response content.
