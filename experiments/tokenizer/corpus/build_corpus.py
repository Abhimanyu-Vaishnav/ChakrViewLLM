"""
ChakrView Benchmark Control Corpus Builder (Step 3).

Generates a deterministic 23-category benchmark control corpus in experiments/tokenizer/corpus/.
Categories cover:
A. English (english.txt)
B. Hindi (hindi.txt)
C. Sanskrit (sanskrit.txt)
D. Hinglish (hinglish.txt)
E. Indian names (indian_names.txt)
F. Technical language (technical.txt)
G. Python code (python_code.txt)
H. JSON data (json_data.txt)
I. URLs (urls.txt)
J. Windows paths (windows_paths.txt)
K. Linux paths (linux_paths.txt)
L. Mathematics (mathematics.txt)
M. Numbers (numbers.txt)
N. Currencies (currencies.txt)
O. Dates (dates.txt)
P. Scientific notation (scientific_notation.txt)
Q. Emojis (emojis.txt)
R. Mixed Unicode (mixed_unicode.txt)
S. Devanagari conjuncts (devanagari_conjuncts.txt)
T. ZWJ/ZWNJ (zwj_zwnj.txt)
U. Whitespace (whitespace.txt)
V. CRLF/LF (line_endings.txt)
W. Raw/malformed byte text (raw_bytes.txt)

Generates metadata.json with exact byte counts, line counts, and category descriptions.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

CORPUS_DIR = Path(__file__).parent

# Category Content Definitions
CORPUS_DATA: Dict[str, Dict[str, Any]] = {
    "english.txt": {
        "category": "A_english",
        "description": "Expository English on computing, algorithms, philosophy, and history.",
        "text": """ChakrView is an indigenous artificial intelligence research initiative created to build a lightweight, efficient neural architecture.
The goal of this foundational project is to explore whether useful cognitive abilities can be realized on constrained hardware architectures.
Modern computing systems frequently suffer from memory bandwidth bottlenecks during autoregressive sequence decoding.
When a transformer model evaluates next-token probabilities, every weight tensor must be retrieved from system memory into register caches.
Therefore, reducing the precision of model weights from 32-bit floating point to 8-bit or 4-bit integers directly improves throughput.
Vectorized single-instruction multiple-data (SIMD) instruction sets such as AVX2, AVX-512, and ARM NEON allow parallel arithmetic.
By aligning head dimensions to multiples of vector register widths, inner dot product loops avoid remainder padding overhead.
Language tokenization is the fundamental bridge connecting raw digital text with continuous dimensional vector representations.
A byte-level byte-pair encoding tokenizer guarantees that any sequence of raw octets can be processed without out-of-vocabulary tokens.
Theoretical computer science demonstrates that deterministic algorithms provide reproducible results across different hardware environments.
The balance between vocabulary size, token fertility, and embedding parameter footprint is a critical trade-off in micro-model design.
We seek mathematical clarity, disciplined empirical measurement, and transparent verification before advancing to neural network training.
Knowledge is structured through hierarchical layers of abstraction, from discrete byte tokens to deep relational semantic manifolds.
The CPU cache hierarchy consists of fast L1 instruction and data caches, intermediate L2 caches, and larger shared L3 caches.
Designing architectures that respect memory locality is essential when deploying neural systems on older, edge, or low-cost microprocessors.
""",
    },
    "hindi.txt": {
        "category": "B_hindi",
        "description": "Standard Devanagari Hindi text covering technology, science, history, and literature.",
        "text": """चक्रव्यूह एक स्वदेशी कृत्रिम बुद्धिमत्ता अनुसंधान परियोजना है जिसका उद्देश्य कम संसाधनों वाले कंप्यूटरों पर चलने वाला मस्तिष्क बनाना है।
यह प्रणाली बिना किसी पूर्व-प्रशिक्षित मॉडल या बाहरी भार के पूर्णतः स्वतंत्र रूप से विकसित की जा रही है।
कंप्यूटर विज्ञान में टोकनाइज़र वह माध्यम है जो साधारण मानवीय भाषा को गणितीय संख्याओं में परिवर्तित करता है।
देवनागरी लिपि में स्वर, व्यंजन, मात्राएं और संयुक्त अक्षर भाषा की संरचना को समृद्ध और वैज्ञानिक बनाते हैं।
जब कोई बड़ा भाषा मॉडल वाक्य का निर्माण करता है, तो वह पिछले शब्दों के आधार पर अगले सबसे संभावित शब्द का अनुमान लगाता है।
भारतीय भाषाओं की भाषाई विविधता को डिजिटल दुनिया में समान प्रतिनिधित्व मिलना अति आवश्यक है।
कम मेमोरी और कम बिजली खपत वाले उपकरणों पर भाषा मॉडल चलाना एक महत्वपूर्ण वैज्ञानिक और तकनीकी चुनौती है।
सूचना प्रौद्योगिकी के युग में अपनी भाषा और संस्कृति का संरक्षण तकनीकी आत्मनिर्भरता से ही संभव है।
गणितीय गणनाओं और कलन विधियों की सटीकता किसी भी सफल प्रणाली की आधारशिला होती है।
प्राचीन भारतीय गणितज्ञों जैसे आर्यभट्ट, ब्रह्मगुप्त और भास्कराचार्य ने शून्य और दशमलव प्रणाली का प्रतिपादन किया था।
प्रकृति और विज्ञान के नियमों का गहन अध्ययन हमें नई तकनीकों के विकास की ओर अग्रसर करता है।
स्वदेशी तकनीक का विकास केवल आत्मनिर्भरता ही नहीं बल्कि हमारी बौद्धिक और सांस्कृतिक संप्रभुता का भी प्रतीक है।
""",
    },
    "sanskrit.txt": {
        "category": "C_sanskrit",
        "description": "Classical Sanskrit shlokas, grammatical treatises, sandhi rules, and philosophical sutras.",
        "text": """ॐ सह नाववतु। सह नौ भुनक्तु। सह वीर्यं करवावहै। तेजस्वि नावधीतमस्तु मा विद्विषावहै। ॐ शान्तिः शान्तिः शान्तिः॥
सत्यमेव जयते नानृतं सत्येन पन्था विततो देवयानः। येनाक्रमन्त्यृषयो ह्याप्तकामा यत्र तत् सत्यस्य परमं निधानम्॥
विद्या ददाति विनयं विनयाद् याति पात्रताम्। पात्रत्वाद् धनमाप्नोति धनाद् धर्मं ततः सुखम्॥
अष्टाध्यायी पाणिनेः व्याकरणस्य आधारशिला अस्ति। यत्र माहेश्वरसूत्राणि चतुर्दश सन्ति।
अ इ उ ण्। ऋ ऌ क्। ए ओ ङ्। ऐ औ च्। ह य व र ट्। लँ ण्।
उद्यमेन हि सिध्यन्ति कार्याणि न मनोरथैः। न हि सुप्तस्य सिंहस्य प्रविशन्ति मुखे मृगाः॥
अहिंसा परमो धर्मस्तथाऽहिंसा परं दमः। अहिंसा परमं दानमहिंसा परमं तपः॥
यदा यदा हि धर्मस्य ग्लानिर्भवति भारत। अभ्युत्थानमधर्मस्य तदात्मानं सृजाम्यहम्॥
कर्मण्येवाधिकारस्ते मा फलेषु कदाचन। मा कर्मफलहेतुर्भूर्मा ते सङ्गोऽस्त्वकर्मणि॥
सर्वधर्मान्परित्यज्य मामेकं शरणं व्रज। अहं त्वा सर्वपापेभ्यो मोक्षयिष्यामि मा शुचः॥
""",
    },
    "hinglish.txt": {
        "category": "D_hinglish",
        "description": "Colloquial Romanized Hindi-English code-switching text used in digital communication.",
        "text": """bhai mujhe ek lightweight AI model banana hai jo mere purane laptop par smoothly run kare.
kya ChakrView bina GPU ke sirf CPU par inferencing kar sakta hai?
ha bilkul, iska hidden dimension 192 hai aur context length 512 tokens tak limited hai.
tokenizer ka kaam text ko byte tokens mein split karna hota hai taaki lossless reconstruction possible ho.
agar vocab size 4096 select karein toh embedding matrix ka size 4096 cross 192 hoga.
mujhe lagta hai ki Hindi aur English dono ke liye byte-level BPE sabse best choice hai.
pehle hum synthetic benchmarks par learnability verify karenge fir actual training start hogi.
code compile ho gaya hai aur saare 107 tests green pass ho rahe hain.
aaj hum Step 3 ka empirical tokenizer benchmark execute kar rahe hain taaki measurable data mile.
koi assumption nahi lena hai, sab kuch benchmark evidence se prove karna zaroori hai.
RAM usage aur CPU clock speed optimize karke purane 28nm chips par bhi test karenge.
""",
    },
    "indian_names.txt": {
        "category": "E_indian_names",
        "description": "Proper Indian personal, geographic, cultural, and historical names in English and Devanagari.",
        "text": """Abhimanyu, Aryabhata, Brahmagupta, Bhaskara, Chanakya, Chandragupta, Vikramaditya, Ashoka.
अभिमन्यु, आर्यभट, ब्रह्मगुप्त, भास्कराचार्य, चाणक्य, चन्द्रगुप्त, विक्रमादित्य, अशोक।
Ramanujan, Jagadish Chandra Bose, Satyendra Nath Bose, C. V. Raman, Homi Bhabha, Vikram Sarabhai.
रामानुजन, जगदीश चन्द्र बोस, सत्येन्द्र नाथ बोस, सी वी रमन, होमी भाभा, विक्रम साराभाई।
Ayodhya, Kashi, Prayagraj, Mathura, Haridwar, Ujjain, Dwarka, Puri, Rameswaram, Kedarnath, Badrinath.
अयोध्या, काशी, प्रयागराज, मथुरा, हरिद्वार, उज्जैन, द्वारका, पुरी, रामेश्वरम, केदारनाथ, बद्रीनाथ।
Ganga, Yamuna, Saraswati, Narmada, Godavari, Krishna, Kaveri, Brahmaputra, Sindhu, Jhelum.
गंगा, यमुना, सरस्वती, नर्मदा, गोदावरी, कृष्णा, कावेरी, ब्रह्मपुत्र, सिन्धु, झेलम।
Thiruvananthapuram, Visakhapatnam, Bengaluru, Hyderabad, Chennai, Mumbai, Kolkata, Ahmedabad, Pune.
तिरुवनंतपुरम, विशाखापट्टनम, बेंगलुरु, हैदराबाद, चेन्नई, मुंबई, कोलकाता, अहमदाबाद, पुणे।
""",
    },
    "technical.txt": {
        "category": "F_technical",
        "description": "Technical computer architecture, microprocessor registers, memory hierarchies, and SIMD instruction sets.",
        "text": """The x86-64 microarchitecture implements out-of-order execution pipelines with hardware branch predictors.
Vector processing units support AVX2 256-bit wide registers (ymm0 through ymm15) executing 8 single-precision floats.
AVX-512 extensions provide 512-bit registers (zmm0 through zmm31) doubling vector throughput per instruction cycle.
ARM Cortex-A53 microarchitectures utilize in-order execution pipelines with 128-bit NEON vector processing units.
Memory latency is mitigated through multi-level hardware caching: L1 data cache (32 KB), L2 cache (512 KB), and unified L3 cache.
Direct memory access (DMA) engines transfer contiguous blocks of data without continuous CPU core polling.
Matrix multiplication algorithms optimize inner loops using cache tiling to maximize L1/L2 data residency.
Row-major memory layouts ensure unit-stride memory accesses, preventing translation lookaside buffer (TLB) misses.
Instruction level parallelism (ILP) allows superscalar processors to issue multiple independent micro-ops simultaneously.
Hardware performance counters measure instruction cache misses, branch mispredictions, and memory bus saturation.
""",
    },
    "python_code.txt": {
        "category": "G_python_code",
        "description": "Python source code snippets, classes, decorators, math functions, and data structures.",
        "text": """import math
from typing import Dict, List, Optional, Tuple

class SimpleMultiHeadAttention:
    def __init__(self, d_model: int = 192, n_heads: int = 6):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

    def forward(self, q: List[float], k: List[float], v: List[float]) -> List[float]:
        # Compute scaled dot product attention
        scale = 1.0 / math.sqrt(self.d_head)
        return [val * scale for val in q]

def compute_softmax(logits: List[float]) -> List[float]:
    max_val = max(logits)
    exps = [math.exp(x - max_val) for x in logits]
    sum_exps = sum(exps)
    return [exp_val / sum_exps for exp_val in exps]

@property
def is_deterministic() -> bool:
    return True
""",
    },
    "json_data.txt": {
        "category": "H_json_data",
        "description": "Structured JSON configuration payloads, API responses, and schema manifests.",
        "text": """{
  "project": "ChakrView",
  "version": "0.1.0",
  "architecture": {
    "model_name": "Chakr-Micro",
    "d_model": 192,
    "layers": 6,
    "heads": 6,
    "head_dim": 32,
    "ffn_dim": 512,
    "max_context": 512,
    "vocab_size": 4096,
    "weight_tying": true,
    "bias": false,
    "norm": "RMSNorm",
    "positional_encoding": "RoPE"
  },
  "hardware_targets": [
    "CPU",
    "28nm_class",
    "ARM_NEON",
    "x86_AVX2"
  ],
  "benchmarks": {
    "reconstruction_lossless": true,
    "special_tokens": {
      "<BOS>": 0,
      "<EOS>": 1,
      "<PAD>": 2
    }
  }
}
""",
    },
    "urls.txt": {
        "category": "I_urls",
        "description": "Uniform Resource Locators, query strings, ports, endpoints, and Devanagari path components.",
        "text": """https://chakrview.ai
https://github.com/chakrview/neural-core
http://localhost:8080/api/v1/tokenize?text=hello&bos=true
https://en.wikipedia.org/wiki/Byte_pair_encoding
https://hi.wikipedia.org/wiki/आर्यभट
https://api.chakrview.ai/v0.1/infer?prompt=नमस्ते%20दुनिया&temp=0.7&max_tokens=64
https://subdomain.domain.org/path/to/resource.html#section-3
ftp://mirror.archive.org/datasets/indic/corpus_2026.tar.gz
ws://127.0.0.1:9001/stream?token_id=4095
http://192.168.1.1:80/admin/status
""",
    },
    "windows_paths.txt": {
        "category": "J_windows_paths",
        "description": "Windows file system paths, drive letters, UNC network shares, backslashes, and extensions.",
        "text": """C:\\Users\\abhimanyu\\Project\\ChakrView\\chakrview\\tokenizer\\tokenizer.py
D:\\Project\\ChakrView\\data\\tokenizer_corpus\\english.txt
C:\\Program Files\\Python314\\python.exe
D:\\ChakrView\\experiments\\tokenizer\\reports\\benchmark_results.json
C:\\Windows\\System32\\drivers\\etc\\hosts
\\\\ServerShare\\SharedVolume\\ChakrView\\models\\chakr_micro_v0.1.bin
C:\\Users\\Default User\\AppData\\Local\\Temp\\scratch_test_42.tmp
D:\\Project\\ChakrView\\.venv\\Scripts\\pytest.exe
E:\\Backup\\2026-09-26\\ChakrView_snapshot.zip
C:\\Users\\admin\\Desktop\\research_notes.docx
""",
    },
    "linux_paths.txt": {
        "category": "K_linux_paths",
        "description": "POSIX/Linux file paths, hidden directories, system mount points, and relative paths.",
        "text": """/home/user/Project/ChakrView/chakrview/tokenizer/interface.py
/usr/local/bin/python3
/var/log/chakrview/training_20260926.log
/etc/systemd/system/chakrview-inference.service
/mnt/storage/datasets/indic_corpus/v4096/vocab.json
/opt/intel/mkl/lib/intel64/libmkl_rt.so
~/.config/chakrview/config.toml
./experiments/tokenizer/candidates/v4096/merges.json
../../tests/test_tokenizer_bpe_engine.py
/dev/null
/proc/cpuinfo
""",
    },
    "mathematics.txt": {
        "category": "L_mathematics",
        "description": "Mathematical notation, LaTeX-style expressions, equations, Greek letters, and formulas.",
        "text": """f(x) = (1 / (sigma * sqrt(2 * pi))) * exp(- (x - mu)^2 / (2 * sigma^2))
RoPE(q, m) = [q_0 * cos(m * theta) - q_1 * sin(m * theta), q_0 * sin(m * theta) + q_1 * cos(m * theta)]
RMSNorm(u) = (u / sqrt(mean(u^2) + epsilon)) * gamma
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k) + M) * V
SwiGLU(x) = (Swish(x * W_gate) * (x * W_up)) * W_down
Swish(z) = z * sigmoid(z) = z / (1 + exp(-z))
Loss(theta) = - (1 / N) * sum(log p(y_t | x_{<t}))
sum_{i=1}^{n} i = n * (n + 1) / 2
lim_{x -> 0} (sin(x) / x) = 1
int_{0}^{inf} e^{-x^2} dx = sqrt(pi) / 2
alpha + beta <= gamma * delta
nabla^2 psi + (8 * pi^2 * m / h^2) * (E - V) * psi = 0
""",
    },
    "numbers.txt": {
        "category": "M_numbers",
        "description": "Integers, arbitrary length sequences, phone numbers, counters, and decimal representations.",
        "text": """0 1 2 3 4 5 6 7 8 9
10 11 12 13 14 15 16 17 18 19 20
99 100 101 999 1000 1001 9999 10000
123456789 987654321 112233445566778899
3.14159265358979323846 2.718281828459045
0.000001 0.000000001 99.999%
+91-9876543210 +1-800-555-0199 +44-20-7946-0958
-42 -100 -273.15 +1024 +65536
12:45:59 23:59:59 00:00:00 08:30:15
4096 192 512 6 32 3443136 786432
""",
    },
    "currencies.txt": {
        "category": "N_currencies",
        "description": "Currencies, prices, Indian numbering scale (Lakhs, Crores), and international symbols.",
        "text": """₹50,000 ₹1,00,000 ₹5,00,000 ₹10,00,000 ₹1,50,00,000 ₹100 Crores ₹25 Lakhs
$100.00 $1,000 $10,000 $1,000,000.00 $99.99
€500 €1,250.50 €10,000,000
£75.00 £250,000 £1.5M
¥10,000 ¥1,000,000
₹10 per token, $0.002 per 1k tokens, €0.05 per API call.
Total cost: ₹45,67,890.00 (INR) and $55,000.00 (USD).
The budget allocation is ₹12.5 Crore for indigenous hardware research.
""",
    },
    "dates.txt": {
        "category": "O_dates",
        "description": "Dates, timestamps, ISO 8601 strings, Julian dates, and calendar representations.",
        "text": """2026-09-26 2026-09-26T13:49:21Z 2026-09-26T13:49:21+05:30
26/09/2026 26-09-2026 09/26/2026
Saturday, 26 September 2026
15th August 1947 26th January 1950
2000-01-01 1999-12-31 2024-02-29 2028-02-29
2026-W39-6 2026-269 (Day of year)
From 2026-01-01 to 2026-12-31 inclusive.
Historical era: 322 BCE, 78 CE, 1857 CE, 1947 CE, 2026 CE.
""",
    },
    "scientific_notation.txt": {
        "category": "P_scientific_notation",
        "description": "Scientific notation numbers, exponents, physical constants, and precision specifications.",
        "text": """1.23e-10 6.02214076e+23 2.99792458e8 6.62607015e-34
1.602176634e-19 1.380649e-23 9.1093837e-31 1.67262192e-27
1.0e-5 1.0e-6 1.0e-8 1.0e-4
3.14159e0 -4.56e-3 +7.89e+12
5.670374e-8 8.314462e0 9.80665e0
""",
    },
    "emojis.txt": {
        "category": "Q_emojis",
        "description": "Single codepoint emojis, multi-codepoint sequences, skin tone modifiers, flags, and tech symbols.",
        "text": """🚀 ❤️ 🔥 👍 💻 🧠 🇮🇳 ⚡ 🌐 🎯
👨👩👧👦 (Family sequence: man, woman, girl, boy)
👩💻 (Woman technologist sequence)
👍🏽 (Thumbs up with medium skin tone modifier)
🇮🇳 (Flag of India regional indicator pair)
🏳️‍🌈 (Rainbow flag with ZWJ)
🤖 Indigenous AI brain 🧠 running on CPU 💻 with maximum efficiency ⚡
🎉 107 tests passing cleanly! 🏆
☀️ 🌙 ⭐ 🌧️ ❄️
""",
    },
    "mixed_unicode.txt": {
        "category": "R_mixed_unicode",
        "description": "Multi-script polyglot sentences interweaving Hindi, Sanskrit, English, Latin, and special symbols.",
        "text": """ChakrView (चक्रव्यूह) v0.1: Indigenous AI language model for low-resource CPUs.
The Sanskrit aphorism 'सत्यमेव जयते' translates to 'Truth alone triumphs'.
In 2026, we test ₹50,000 edge hardware using Python 3.14 on Windows 11.
नमस्ते (Namaste) -> Hello -> Bonjour -> Hola -> Привет -> مرحبا -> こんにちは.
Mathematical identity: ∀x ∈ ℝ, e^{iπ} + 1 = 0.
Devanagari numerals: ०, १, २, ३, ४, ५, ६, ७, ८, ९ vs Western: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9.
Testing combining characters: a + ̈ = ä, e + ́ = é, o + ̂ = ô.
Mixed currency transaction: ₹1,00,000 INR equivalent to $1,200 USD and €1,100 EUR.
""",
    },
    "devanagari_conjuncts.txt": {
        "category": "S_devanagari_conjuncts",
        "description": "Complex Devanagari conjuncts, consonant clusters, virama combinations, and ligatures.",
        "text": """कौशाम्बी त्र्यंबकेश्वर कुंडलियाँ पूँछ अँधेरा ऋग्वेद ॐ
सत्त्व दग्ध बुद्ध उष्ट्र कार्त्तिकेय वाग्देवी
क्षत्रिय त्रिशूल ज्ञान प्रकाश श्रम
द्वंद्व बुद्धि विद्या स्वास्थ्य आश्चर्य
उज्ज्वल निष्ठा प्रतिष्ठा संकल्प सिद्धांत
दृष्टिकोण प्रज्ञान चक्रव्यूह आत्मनिर्भर अनुसंधान
स्मृति श्रद्धा प्रतिष्ठा श्लोक राष्ट्र
""",
    },
    "zwj_zwnj.txt": {
        "category": "T_zwj_zwnj",
        "description": "Explicit zero-width joiner (U+200D) and zero-width non-joiner (U+200C) ligature sequences.",
        "text": """क्\u200Dष (Half-ka + ssa with ZWJ: क्‍ष)
क्\u200Cष (Virama-ka + ssa with ZWNJ: क्‌ष)
श्री\u200Dमान् (Shri + ZWJ + man: श्री‍मान्)
प\u200Dर (Half-pa + ra: प‍र)
प\u200Cर (Explicit virama pa + ra: प‌र)
\u200D\u200C\u200D\u200C (Isolated alternating ZWJ and ZWNJ characters)
सं\u200Dयोग (Sa + Anusvara + ZWJ + yoga)
नि\u200Cयम (Ni + ZWNJ + yama)
""",
    },
    "whitespace.txt": {
        "category": "U_whitespace",
        "description": "Single spaces, multiple spaces, tabs, mixed indents, leading, and trailing whitespaces.",
        "text": """Single space between words.
Two  spaces   three    spaces     four      spaces.
\tSingle tab indented line.
\t\tDouble tab indented line.
    Four spaces indentation.
        Eight spaces indentation.
Mixed \t space \t and \t tab.
No_spaces_at_all_here.
   Leading spaces on this line.
Trailing spaces on this line.   
   Leading and trailing spaces.   
\t\t\tMultiple tabs only.
""",
    },
    "line_endings.txt": {
        "category": "V_line_endings",
        "description": "Windows CRLF, Unix LF, and mixed carriage return and line feed variations.",
        "text": "CRLF_Line_1\r\nCRLF_Line_2\r\nCRLF_Line_3\r\n"
        "LF_Line_1\nLF_Line_2\nLF_Line_3\n"
        "Mixed_Line_1\r\nMixed_Line_2\nMixed_Line_3\r\n"
        "Consecutive_Blank_Lines\r\n\r\n\n\n\r\nEnd_Of_Lines\n",
    },
    "raw_bytes.txt": {
        "category": "W_raw_bytes",
        "description": "Text representation documenting raw byte primitives, escape codes, and byte boundary targets.",
        "text": """Raw byte 0x00 is NULL byte, encoded as foundational token ID 3.
Raw byte 0xFF is 255, encoded as foundational token ID 258.
Byte 0x0A is newline, encoded as foundational token ID 13.
Byte 0x20 is space, encoded as foundational token ID 35.
Devanagari characters are 3-byte UTF-8 sequences starting with 0xE0 0xA4 or 0xE0 0xA5.
Emojis are 4-byte UTF-8 sequences starting with 0xF0 0x9F.
High bytes 0x80 through 0xFF without UTF-8 header are preserved as individual byte tokens.
No matter what arbitrary 8-bit octet sequence is provided, byte-level BPE reconstructs it bit-for-bit.
All 256 fundamental byte primitives map bijectively to token IDs 3 through 258 without collision.
""",
    },
}


def build_control_corpus() -> Dict[str, Any]:
    """Write all 23 category files and generate metadata.json."""
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    metadata: Dict[str, Any] = {
        "name": "ChakrView_Benchmark_Control_Corpus_v0.1",
        "total_categories": len(CORPUS_DATA),
        "categories": {},
    }

    total_bytes = 0
    total_lines = 0

    for filename, data in sorted(CORPUS_DATA.items()):
        filepath = CORPUS_DIR / filename
        text = data["text"]
        filepath.write_text(text, encoding="utf-8")

        raw_bytes = text.encode("utf-8")
        lines = text.splitlines()
        sha256 = hashlib.sha256(raw_bytes).hexdigest()

        cat_meta = {
            "category": data["category"],
            "filename": filename,
            "description": data["description"],
            "byte_count": len(raw_bytes),
            "line_count": len(lines),
            "sha256": sha256,
        }
        metadata["categories"][filename] = cat_meta
        total_bytes += len(raw_bytes)
        total_lines += len(lines)

    metadata["total_bytes"] = total_bytes
    metadata["total_lines"] = total_lines

    meta_file = CORPUS_DIR / "metadata.json"
    with meta_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Built control corpus: {len(CORPUS_DATA)} categories, {total_lines} lines, {total_bytes} UTF-8 bytes.")
    return metadata


if __name__ == "__main__":
    build_control_corpus()
