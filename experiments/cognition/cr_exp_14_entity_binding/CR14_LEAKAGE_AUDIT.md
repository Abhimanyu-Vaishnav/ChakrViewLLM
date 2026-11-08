# CR-EXP-14 BENCHMARK LEAKAGE & INTEGRITY AUDIT
**Auditor:** Autonomous Scientific Integrity Auditor  
**Date:** 2026-11-08T10:11:50+05:30  
**Benchmark:** `benchmark_1200_sealed.json` (`1958e94571f83c66c26f390634ffa563d0a2ff57e58943e835d7825245b964b9`)  

---

## 1. ENTITY INVENTORY DISJOINTNESS AUDIT
- **Benchmark Names ($N=40$)**:
  `Abhay`, `Bimla`, `Chetan`, `Damodar`, `Ekta`, `Girish`, `Hema`, `Indu`, `Janak`, `Kamlesh`, `Laxman`, `Mina`, `Narayan`, `Omwati`, `Pushpa`, `Ratan`, `Sarla`, `Trilok`, `Urmila`, `Ved`, `Aniruddh`, `Bela`, `Chirag`, `Daksha`, `Gajendra`, `Hansmukh`, `Jagruti`, `Kailashpati`, `Lata`, `Mukund`, `Nalini`, `Ojasvi`, `Prakash`, `Radha`, `Subhash`, `Trupti`, `Udit`, `Vatsala`, `Yadunath`, `Zulekha`.
- **CR-11 & CR-12 Training Names ($N=40$)**:
  `Aarav`, `Meera`, `Kabir`, `Neha`, `Rohan`, `Ananya`, `Vikram`, `Pooja`, `Aditya`, `Sneha`, `Kunal`, `Tanvi`, `Siddharth`, `Ishita`, `Varun`, `Riya`, `Gaurav`, `Kavya`, `Deepak`, `Simran`, `Naveen`, `Swati`, `Rahul`, `Divya`, `Amit`, `Shreya`, `Manish`, `Preeti`, `Sanjay`, `Ankita`, `Alok`, `Priya`, `Harsh`, `Karan`, `Payal`, `Sachin`, `Archana`, `Mayank`, `Tarun`, `Komal`.
- **CR-12 Benchmark Names ($N=40$)**:
  `Aalok`, `Brijesh`, `Charu`, `Damini`, `Eshwar`, `Farhan`, etc.
- **Overlap Audit**:
  $$\text{Benchmark Names} \cap \text{Historical Training Names} = \emptyset$$
  $$\text{Benchmark Names} \cap \text{CR-12 Benchmark Names} = \emptyset$$
  **Status**: **100% DISJOINT AND ZERO LEAKAGE CONFIRMED**.

---

## 2. PROMPT DUPLICATION & SYNTACTIC ISOLATION
- Zero duplicate prompts exist in the benchmark.
- Dialogue templates use diverse framing variations (`"Please note that X..."`, `"Tracking group: X in A..."`, `"X arrived at A..."`).
- Test probes are permanently sealed and strictly forbidden from training access.
