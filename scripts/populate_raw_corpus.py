"""
ChakrView Raw Corpus Generator (Step 3).

Populates data/raw/ with high-density, multi-domain, indigenous text covering:
- data/raw/hindi/ (~35%)
- data/raw/english/ (~25%)
- data/raw/hinglish/ (~15%)
- data/raw/sanskrit/ (~10%)
- data/raw/mixed/ (~10%)
- data/raw/code/ (~10%)
- data/raw/mathematics/ (~5%)
- data/raw/numbers/ (~5%)

Generates data/raw/manifest.json containing exact metadata:
- source
- license / status (all permissively licensed, authored, or public domain)
- language
- category
- creation method
- preprocessing status
- file hashes (SHA-256)
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# Comprehensive linguistic corpus definitions
CORPUS_CONTENT = {
    "hindi": {
        "hindi_computing.txt": (
            "चक्रव्यूह एक स्वदेशी कृत्रिम बुद्धिमत्ता अनुसंधान परियोजना है जिसका उद्देश्य कम संसाधनों वाले कंप्यूटरों पर चलने वाला तंत्रिका मस्तिष्क विकसित करना है।\n"
            "यह प्रणाली बिना किसी पूर्व-प्रशिक्षित मॉडल या विदेशी सॉफ्टवेयर भार के पूर्णतः स्वतंत्र रूप से बनाई जा रही है।\n"
            "कंप्यूटर विज्ञान में टोकनाइज़र वह महत्वपूर्ण माध्यम है जो साधारण मानवीय भाषा को गणितीय संख्याओं और वेक्टरों में परिवर्तित करता है।\n"
            "देवनागरी लिपि में स्वर, व्यंजन, मात्राएं और संयुक्त अक्षर भाषा की संरचना को अत्यंत समृद्ध और वैज्ञानिक बनाते हैं।\n"
            "जब कोई बड़ा भाषा मॉडल वाक्य का निर्माण करता है, तो वह पिछले शब्दों के आधार पर अगले सबसे संभावित शब्द का अनुमान लगाता है।\n"
            "भारतीय भाषाओं की भाषाई विविधता को डिजिटल दुनिया में समान और प्रभावी प्रतिनिधित्व मिलना अति आवश्यक है।\n"
            "कम मेमोरी और कम बिजली खपत वाले उपकरणों पर भाषा मॉडल चलाना एक महत्वपूर्ण वैज्ञानिक और तकनीकी चुनौती है।\n"
            "सूचना प्रौद्योगिकी के युग में अपनी भाषा और संस्कृति का संरक्षण तकनीकी आत्मनिर्भरता से ही संभव हो सकता है।\n"
            "गणितीय गणनाओं और कलन विधियों की सटीकता किसी भी सफल कम्प्यूटेशनल प्रणाली की मजबूत आधारशिला होती है।\n"
            "प्राचीन भारतीय गणितज्ञों जैसे आर्यभट्ट, ब्रह्मगुप्त और भास्कराचार्य ने शून्य और दशमलव प्रणाली का प्रतिपादन किया था।\n"
            "प्रकृति और विज्ञान के नियमों का गहन अध्ययन हमें नई तकनीकों के अनुसंधान और विकास की ओर अग्रसर करता है।\n"
            "स्वदेशी तकनीक का निर्माण केवल आत्मनिर्भरता ही नहीं बल्कि हमारी बौद्धिक और तकनीकी संप्रभुता का भी प्रतीक है।\n"
            "कृत्रिम बुद्धिमत्ता के मॉडल को प्रशिक्षित करने के लिए उच्च गुणवत्ता वाले डेटासेट और संतुलित शब्दावली की आवश्यकता होती है।\n"
            "ऑटोरेग्रेसिव डिकोडर भाषा मॉडल प्रत्येक चरण पर एक टोकन उत्पन्न करता है और उसे पिछले संदर्भ के साथ जोड़ता है।\n"
            "समानांतर कंप्यूटिंग के लिए सीपीयू के एसआईएमडी रजिस्टरों का कुशल उपयोग मेमोरी बैंडविड्थ की खपत को कम करता है।\n"
        ),
        "hindi_science_culture.txt": (
            "विज्ञान और प्रौद्योगिकी का मुख्य उद्देश्य मानव जीवन को सरल, समृद्ध और ज्ञानवर्धक बनाना है।\n"
            "भारत का खगोल विज्ञान और गणित का इतिहास अत्यंत प्राचीन और गौरवशाली रहा है।\n"
            "सूर्य सिद्धांत, खगोल शास्त्र और त्रिकोणमिति में भारतीय विद्वानों के योगदान को विश्वभर में मान्यता प्राप्त है।\n"
            "भाषा केवल संवाद का साधन नहीं है, बल्कि यह संस्कृति, दर्शन और मानवीय सोच की वाहक भी है।\n"
            "आधुनिक समय में डिजिटल संचार के विस्तार के कारण विभिन्न भाषाओं का एक दूसरे के साथ सम्मिश्रण तेजी से बढ़ रहा है।\n"
            "शिक्षा में मातृभाषा का प्रयोग बच्चों के मानसिक विकास और रचनात्मक सोच को बढ़ावा देता है।\n"
            "कंप्यूटर और मोबाइल फोन में देवनागरी लिपि के सुगम प्रयोग के लिए यूनिकोड मानकीकरण ने महत्वपूर्ण भूमिका निभाई है।\n"
            "कौशाम्बी, त्र्यंबकेश्वर, कुंडलियाँ, पूँछ, अँधेरा और ऋग्वेद जैसे जटिल शब्द देवनागरी की विशिष्टता को दर्शाते हैं।\n"
            "संयुक्त व्यंजनों का निर्माण हलंत के माध्यम से होता है, जहां दो या दो से अधिक व्यंजन मिलकर एक नया रूप धारण करते हैं।\n"
            "प्रकृति के संरक्षण और सतत विकास के लिए वैज्ञानिक अनुसंधान और पारंपरिक ज्ञान का समन्वय अत्यंत आवश्यक है।\n"
            "कृत्रिम तंत्रिका नेटवर्क मानव मस्तिष्क के न्यूरॉन्स की कार्यप्रणाली से प्रेरित होकर कार्य करते हैं।\n"
            "स्मृति, ध्यान और तर्क करने की क्षमता किसी भी बुद्धिमान संज्ञानात्मक प्रणाली के प्रमुख घटक होते हैं।\n"
            "स्वदेशी अनुसंधान परियोजनाओं से देश के युवा वैज्ञानिकों और इंजीनियरों को नए अवसर प्राप्त होते हैं।\n"
            "डिजिटल युग में सूचना की सुरक्षा और डेटा गोपनीयता संप्रभुता के महत्वपूर्ण पहलू बन चुके हैं।\n"
        ),
        "hindi_dialogue.txt": (
            "नमस्ते, आप आज कैसे हैं? मैं आपकी क्या सहायता कर सकता हूँ?\n"
            "मुझे कंप्यूटर प्रोग्रामिंग और मशीन लर्निंग के बारे में बुनियादी जानकारी चाहिए।\n"
            "निश्चय ही! प्रोग्रामिंग कंप्यूटर को निर्देश देने की एक व्यवस्थित कला है।\n"
            "पायथन भाषा सीखने में बहुत सरल है और डेटा विज्ञान में इसका सर्वाधिक उपयोग किया जाता है।\n"
            "क्या हम कम क्षमता वाले लैपटॉप पर भी एआई मॉडल का परीक्षण कर सकते हैं?\n"
            "हाँ, बिल्कुल! चक्रव्यूह का माइक्रो मॉडल विशेष रूप से साधारण सीपीयू के लिए ही डिज़ाइन किया गया है।\n"
            "इसमें केवल 34 लाख पैरामीटर हैं और यह बहुत कम रैम में भी तीव्र गति से काम करता है।\n"
            "टोकनाइजेशन की प्रक्रिया में प्रत्येक शब्द या उप-शब्द को एक अद्वितीय पहचान संख्या दी जाती है।\n"
            "इससे मॉडल भाषा के व्याकरण और शब्दों के आपसी संबंधों को आसानी से समझ सकता है।\n"
            "आपका बहुत-बहुत धन्यवाद, यह जानकारी मेरे लिए अत्यंत उपयोगी और प्रेरणादायक है।\n"
            "शुभकामनाएं! निरंतर अभ्यास और अध्ययन से आप इस क्षेत्र में उत्कृष्ट सफलता प्राप्त करेंगे।\n"
        ),
    },
    "english": {
        "english_systems.txt": (
            "ChakrView is an indigenous artificial intelligence research initiative created to build a lightweight neural architecture.\n"
            "The overarching objective of this foundational project is to explore whether cognitive abilities can be realized on constrained hardware.\n"
            "Modern computing systems frequently encounter severe memory bandwidth bottlenecks during autoregressive sequence decoding.\n"
            "When a transformer model evaluates next-token probabilities, every weight tensor must be retrieved from main memory into registers.\n"
            "Therefore, quantizing model weights from 32-bit floating point to 8-bit or 4-bit integers directly improves operational throughput.\n"
            "Vectorized single-instruction multiple-data (SIMD) instruction sets such as AVX2, AVX-512, and ARM NEON allow parallel arithmetic.\n"
            "By aligning attention head dimensions to multiples of vector register widths, matrix multiplication loops eliminate padding overhead.\n"
            "Language tokenization forms the critical mathematical bridge connecting raw digital text with continuous embedding manifolds.\n"
            "A byte-level byte-pair encoding tokenizer guarantees that any sequence of raw octets can be processed without out-of-vocabulary tokens.\n"
            "Theoretical computer science demonstrates that deterministic algorithms yield reproducible results across diverse hardware platforms.\n"
            "The trade-off between vocabulary size, token fertility, and embedding parameter footprint is a central axis of micro-model design.\n"
            "We prioritize mathematical rigor, transparent verification, and empirical measurement before commencing neural network training.\n"
            "Knowledge is represented through hierarchical computational transformations, from discrete byte tokens to deep associative manifolds.\n"
            "The processor cache hierarchy consists of high-speed L1 instruction and data caches, intermediate L2 caches, and shared L3 caches.\n"
            "Designing neural architectures that respect cache locality is vital when deploying models on legacy or low-cost microprocessors.\n"
        ),
        "english_computer_science.txt": (
            "Data structures and algorithms form the bedrock of computational efficiency and software performance engineering.\n"
            "A hash table provides expected constant-time operations for lookups, insertions, and deletions under uniform hashing assumptions.\n"
            "Binary search trees and balanced red-black trees guarantee logarithmic worst-case search times for ordered collections of keys.\n"
            "In natural language processing, the byte-pair encoding algorithm iteratively replaces the most frequent byte pairs with novel tokens.\n"
            "This compression technique balances sequence length reduction against the memory required to store the vocabulary lookup table.\n"
            "Linear algebra routines implemented in BLAS libraries utilize loop unrolling, register blocking, and cache tiling to saturate hardware.\n"
            "Memory-bound operations exhibit low arithmetic intensity, meaning the processor spends more cycles waiting for memory than computing.\n"
            "Compute-bound operations, by contrast, keep the arithmetic logic units fully saturated with matrix multiplication operations.\n"
            "Operating systems manage virtual memory through page tables, translating virtual addresses to physical RAM addresses via the MMU.\n"
            "Cache misses and translation lookaside buffer (TLB) evictions introduce substantial latency penalties in memory-intensive workloads.\n"
            "Compiler optimizations such as instruction reordering, dead code elimination, and auto-vectorization improve machine code efficiency.\n"
            "Deterministic systems engineering ensures that identical inputs and configurations generate bit-exact identical execution outputs.\n"
        ),
    },
    "hinglish": {
        "hinglish_tech.txt": (
            "bhai mujhe ek lightweight indigenous AI model develop karna hai jo low-spec CPU par smoothly run kare.\n"
            "kya ChakrView bina discrete GPU ke sirf Intel ya AMD processor par inferencing kar sakta hai?\n"
            "ha bilkul, iska hidden dimension 192 hai aur context length 512 tokens tak tightly bound hai.\n"
            "tokenizer ka kaam text ko byte tokens mein split karna hota hai taaki lossless reconstruction guarantee ho sake.\n"
            "agar vocab size 4096 select karein toh embedding matrix ka exact size 4096 cross 192 parameters hoga.\n"
            "mujhe lagta hai ki Hindi aur English dono ke liye byte-level BPE sabse practical aur balanced choice hai.\n"
            "pehle hum synthetic benchmarks par learnability verify karenge uske baad actual language pre-training start hogi.\n"
            "code test suite mein saare unit tests completely green pass ho rahe hain bina kisi assertion failure ke.\n"
            "aaj hum Step 3 ka empirical tokenizer benchmark execute kar rahe hain taaki measurable evidence collect ho.\n"
            "koi bhi architectural assumption guess work par nahi lena hai, sab kuch benchmark numbers se justify hona chahiye.\n"
            "RAM consumption aur CPU clock cycles optimize karke hum purane 28nm processors par bhi testing perform karenge.\n"
            "yeh project open source principles aur indigenous technology ki foundational philosophy par based hai.\n"
            "agar quantization implement karein toh INT8 weights ka memory footprint lagbhag 3.44 MB tak reduce ho jayega.\n"
            "kisi bhi modern neural network mein attention mechanism token-to-token relationships ko deeply capture karta hai.\n"
        ),
    },
    "sanskrit": {
        "sanskrit_classics.txt": (
            "ॐ सह नाववतु। सह नौ भुनक्तु। सह वीर्यं करवावहै। तेजस्वि नावधीतमस्तु मा विद्विषावहै। ॐ शान्तिः शान्तिः शान्तिः॥\n"
            "सत्यमेव जयते नानृतं सत्येन पन्था विततो देवयानः। येनाक्रमन्त्यृषयो ह्याप्तकामा यत्र तत् सत्यस्य परमं निधानम्॥\n"
            "विद्या ददाति विनयं विनयाद् याति पात्रताम्। पात्रत्वाद् धनमाप्नोति धनाद् धर्मं ततः सुखम्॥\n"
            "अष्टाध्यायी पाणिनेः व्याकरणस्य आधारशिला अस्ति। यत्र माहेश्वरसूत्राणि चतुर्दश सन्ति।\n"
            "अ इ उ ण्। ऋ ऌ क्। ए ओ ङ्। ऐ औ च्। ह य व र ट्। लँ ण्। ञ म ङ ण न म्। झ भ ञ्। घ ढ ध ष्। ज ब ग ड द श्। ख फ छ ठ थ च ट त व्। क प य्। श ष स र्। ह ल्।\n"
            "उद्यमेन हि सिध्यन्ति कार्याणि न मनोरथैः। न हि सुप्तस्य सिंहस्य प्रविशन्ति मुखे मृगाः॥\n"
            "अहिंसा परमो धर्मस्तथाऽहिंसा परं दमः। अहिंसा परमं दानमहिंसा परमं तपः॥\n"
            "यदा यदा हि धर्मस्य ग्लानिर्भवति भारत। अभ्युत्थानमधर्मस्य तदात्मानं सृजाम्यहम्॥\n"
            "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन। मा कर्मफलहेतुर्भूर्मा ते सङ्गोऽस्त्वकर्मणि॥\n"
            "सर्वधर्मान्परित्यज्य मामेकं शरणं व्रज। अहं त्वा सर्वपापेभ्यो मोक्षयिष्यामि मा शुचः॥\n"
            "सत्त्वं रजस्तम इति गुणाः प्रकृतिसंभवाः। निबध्नन्ति महाबाहो देहे देहिनमव्ययम्॥\n"
            "दग्धं बीजं न रोहति। बुद्धस्य वचनं कल्याणकारी। उष्ट्रः मरुभूमौ भ्रमति। कार्त्तिकेयः सेनापतिः। वाग्देवी भारती नमस्कृत्य।\n"
        ),
    },
    "mixed": {
        "mixed_multilingual.txt": (
            "ChakrView (चक्रव्यूह) v0.1: Indigenous AI language model designed specifically for low-resource CPUs.\n"
            "The sacred Sanskrit aphorism 'सत्यमेव जयते' translates to 'Truth alone triumphs' in modern English.\n"
            "In 2026, we test ₹50,000 edge hardware using Python 3.14 on Windows 11 and Ubuntu Linux.\n"
            "नमस्ते (Namaste) -> Hello -> Bonjour -> Hola -> Привет -> مرحبا -> こんにちは.\n"
            "Mathematical identity: ∀x ∈ ℝ, e^{iπ} + 1 = 0, combining analysis, geometry, and arithmetic.\n"
            "Devanagari numerals: ०, १, २, ३, ४, ५, ६, ७, ८, ९ compared with Western Arabic numerals: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9.\n"
            "Testing combining characters: a + ̈ = ä, e + ́ = é, o + ̂ = ô with lossless byte-level reconstruction.\n"
            "Mixed currency transaction: ₹1,00,000 INR equivalent to approximately $1,200 USD and €1,100 EUR.\n"
            "Complex Devanagari test: कौशाम्बी, त्र्यंबकेश्वर, कुंडलियाँ, पूँछ, अँधेरा, ऋग्वेद, ॐ.\n"
            "Classical Sanskrit conjuncts: सत्त्व, दग्ध, बुद्ध, उष्ट्र, कार्त्तिकेय, वाग्देवी.\n"
            "Zero-width characters: क्\u200Dष (half-ka with ZWJ) versus क्\u200Cष (virama with ZWNJ).\n"
            "Emoji sequences: 👨👩👧👦 (family), 👩💻 (technologist), 👍🏽 (skin tone), 🇮🇳 (national flag).\n"
        ),
    },
    "code": {
        "code_python.txt": (
            "import math\n"
            "from typing import Dict, List, Optional, Tuple, Final\n\n"
            "class SimpleMultiHeadAttention:\n"
            "    def __init__(self, d_model: int = 192, n_heads: int = 6) -> None:\n"
            "        assert d_model % n_heads == 0, 'd_model must divide evenly by n_heads'\n"
            "        self.d_model = d_model\n"
            "        self.n_heads = n_heads\n"
            "        self.d_head = d_model // n_heads\n\n"
            "    def compute_scaled_scores(self, q: List[float], k: List[float]) -> float:\n"
            "        dot_product = sum(qi * ki for qi, ki in zip(q, k))\n"
            "        scale = 1.0 / math.sqrt(self.d_head)\n"
            "        return dot_product * scale\n\n"
            "def compute_softmax(logits: List[float]) -> List[float]:\n"
            "    max_val = max(logits)\n"
            "    exps = [math.exp(x - max_val) for x in logits]\n"
            "    sum_exps = sum(exps)\n"
            "    return [val / sum_exps for val in exps]\n\n"
            "def calculate_entropy(probabilities: List[float]) -> float:\n"
            "    return -sum(p * math.log2(p) for p in probabilities if p > 0.0)\n\n"
            "# Assert architectural invariants\n"
            "assert SimpleMultiHeadAttention(192, 6).d_head == 32\n"
        ),
        "code_configs.txt": (
            "{\n"
            '  "project": "ChakrView",\n'
            '  "version": "0.1.0",\n'
            '  "architecture": {\n'
            '    "model_name": "Chakr-Micro",\n'
            '    "d_model": 192,\n'
            '    "layers": 6,\n'
            '    "heads": 6,\n'
            '    "head_dim": 32,\n'
            '    "ffn_dim": 512,\n'
            '    "max_context": 512,\n'
            '    "vocab_size": 4096,\n'
            '    "weight_tying": true,\n'
            '    "bias": false,\n'
            '    "norm": "Pre-RMSNorm",\n'
            '    "positional_encoding": "RoPE"\n'
            "  },\n"
            '  "special_tokens": {\n'
            '    "<BOS>": 0,\n'
            '    "<EOS>": 1,\n'
            '    "<PAD>": 2\n'
            "  }\n"
            "}\n"
        ),
    },
    "mathematics": {
        "math_formulas.txt": (
            "f(x) = (1 / (sigma * sqrt(2 * pi))) * exp(- (x - mu)^2 / (2 * sigma^2))\n"
            "RoPE(q, m) = [q_0 * cos(m * theta) - q_1 * sin(m * theta), q_0 * sin(m * theta) + q_1 * cos(m * theta)]\n"
            "RMSNorm(u) = (u / sqrt(mean(u^2) + epsilon)) * gamma\n"
            "Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k) + M) * V\n"
            "SwiGLU(x) = (Swish(x * W_gate) * (x * W_up)) * W_down\n"
            "Swish(z) = z * sigmoid(z) = z / (1 + exp(-z))\n"
            "Loss(theta) = - (1 / N) * sum(log p(y_t | x_{<t}))\n"
            "sum_{i=1}^{n} i = n * (n + 1) / 2\n"
            "lim_{x -> 0} (sin(x) / x) = 1\n"
            "int_{0}^{inf} e^{-x^2} dx = sqrt(pi) / 2\n"
            "alpha + beta <= gamma * delta\n"
            "nabla^2 psi + (8 * pi^2 * m / h^2) * (E - V) * psi = 0\n"
            "e^{i * theta} = cos(theta) + i * sin(theta)\n"
            "det(A * B) = det(A) * det(B)\n"
        ),
    },
    "numbers": {
        "numbers_dataset.txt": (
            "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20\n"
            "99 100 101 999 1000 1001 9999 10000 2026 2027 2030\n"
            "123456789 987654321 112233445566778899 1234567890\n"
            "3.14159265358979323846 2.718281828459045 1.41421356237\n"
            "0.000001 0.000000001 99.999% 12.5% 0.05% -42 -100 -273.15 +1024 +65536\n"
            "1.23e-10 6.022e+23 2.99792e8 6.626e-34 1.602e-19\n"
            "+91-9876543210 +1-800-555-0199 +44-20-7946-0958\n"
            "₹50000 ₹1,00,000 ₹10,00,000 $1000 $99.99 €500 £250 ¥10000\n"
            "2026-09-26 2026-09-26T14:04:22+05:30 12:45:59 23:59:59 00:00:00\n"
            "192.168.1.1 127.0.0.1 10.0.0.1 255.255.255.0\n"
            "v0.1.0 v1.0.0-rc1 v2.4.5\n"
            "123 + 456 = 579\n"
            "9999 * 8888 = 88871112\n"
            "(123 + 456) / 7 = 82.7142857\n"
        ),
    },
}

# Incorporate foundational project-authored corpus files from data/tokenizer_corpus
TOKENIZER_CORPUS_DIR = Path(__file__).resolve().parents[1] / "data" / "tokenizer_corpus"
if TOKENIZER_CORPUS_DIR.is_dir():
    foundation_map = {
        "hindi": ("hindi.txt", "hindi_foundation.txt"),
        "english": ("english.txt", "english_foundation.txt"),
        "hinglish": ("hinglish.txt", "hinglish_foundation.txt"),
        "code": ("code.txt", "code_foundation.txt"),
        "mathematics": ("math.txt", "math_foundation.txt"),
        "numbers": ("numbers.txt", "numbers_foundation.txt"),
        "mixed": ("mixed.txt", "mixed_foundation.txt"),
    }
    for cat, (src_name, dst_name) in foundation_map.items():
        src_path = TOKENIZER_CORPUS_DIR / src_name
        if src_path.is_file():
            CORPUS_CONTENT.setdefault(cat, {})[dst_name] = src_path.read_text(encoding="utf-8")


def build_raw_corpus() -> Dict[str, Any]:
    """Write all raw files with category subdirectories and generate manifest."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        "title": "ChakrView Raw Benchmark Corpus (Step 3)",
        "project": "ChakrView",
        "license": "Indigenous / Public Domain / CC0 Compatible",
        "description": "Controlled, reproducible multi-category text corpus for empirical tokenizer evaluation",
        "categories": {},
        "total_files": 0,
        "total_bytes": 0,
    }

    total_bytes = 0
    total_files = 0
    category_byte_totals: Dict[str, int] = {}

    for cat_name, file_dict in sorted(CORPUS_CONTENT.items()):
        cat_dir = RAW_DIR / cat_name
        cat_dir.mkdir(parents=True, exist_ok=True)
        manifest["categories"][cat_name] = {"files": [], "category_bytes": 0}
        cat_bytes = 0

        for fname, text in sorted(file_dict.items()):
            fpath = cat_dir / fname
            fpath.write_text(text, encoding="utf-8")
            raw_b = text.encode("utf-8")
            sha256 = hashlib.sha256(raw_b).hexdigest()
            b_count = len(raw_b)
            line_count = len(text.splitlines())

            file_meta = {
                "filename": fname,
                "relative_path": f"{cat_name}/{fname}",
                "category": cat_name,
                "language": cat_name,
                "byte_count": b_count,
                "line_count": line_count,
                "sha256": sha256,
                "source": "Project-authored, indigenous synthesis & public domain classical texts",
                "license": "CC0-1.0 / MIT Compatible",
                "preprocessing_status": "Clean UTF-8 raw text without destructive normalization",
            }
            manifest["categories"][cat_name]["files"].append(file_meta)
            cat_bytes += b_count
            total_bytes += b_count
            total_files += 1

        manifest["categories"][cat_name]["category_bytes"] = cat_bytes
        category_byte_totals[cat_name] = cat_bytes

    manifest["total_files"] = total_files
    manifest["total_bytes"] = total_bytes
    manifest["category_distributions"] = {
        cat: f"{round(b / total_bytes * 100, 2)}%" for cat, b in category_byte_totals.items()
    }

    manifest_path = RAW_DIR / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Generated Raw Corpus: {total_files} files, {total_bytes:,} UTF-8 bytes.")
    for cat, pct in manifest["category_distributions"].items():
        print(f"  - {cat:<12}: {category_byte_totals[cat]:>6,} bytes ({pct})")

    return manifest


if __name__ == "__main__":
    build_raw_corpus()
