# CR-EXP-15 Phase 3: Zero Data Leakage Audit

## 1. Audit Scope & Verification
This audit verifies that the 1,200 frozen probes in `benchmark_data/benchmark_1200_sealed.json` (SHA-256: `71212797975022ad4382d4d660d08125782ae473b85a5081eb2f12d7da5d28d8`) are completely disjoint from all past and current training datasets.

---

## 2. Set Disjointness Verification

### A. Entity Isolation
- **Benchmark Entities ($N=40$):** `Apurva`, `Badri`, `Bhargav`, `Chaitanya`, `Devendra`, `Digvijay`, `Eshwar`, `Fateh`, `Gautam`, `Govind`, `Harish`, `Himanshu`, `Indrajit`, `Jaideep`, `Jawahar`, `Kalyan`, `Kuldeep`, `Lalit`, `Madhav`, `Manvendra`, `Mukesh`, `Naveen`, `Niraj`, `Padma`, `Pankhuri`, `Pranav`, `Raghav`, `Rajnath`, `Ranveer`, `Sanjay`, `Sarvesh`, `Shashank`, `Siddharth`, `Somesh`, `Tarun`, `Umesh`, `Vaibhav`, `Varun`, `Vikas`, `Yashwant`.
- **Training Pool Entities:** Excluded from all training sets. All training sets generated in CR-15 must sample strictly from disjoint name lists (`Aaditya`, `Aarti`, `Amrita`, `Aniket`, etc.).
- **Intersection:** $\emptyset$ (0 names).

### B. Location Isolation
- **Benchmark Locations ($N=25$):** `Alwar`, `Bareilly`, `Belgaum`, `Calicut`, `Durgapur`, `Faridabad`, `Gorakhpur`, `Haridwar`, `Imphal`, `Jamnagar`, `Kakinada`, `Kollam`, `Malda`, `Muzaffarpur`, `Nellore`, `Pathankot`, `Rajahmundry`, `Rourkela`, `Saharanpur`, `Thanjavur`, `Thrissur`, `Udaipur`, `Ujjain`, `Vellore`, `Warangal`.
- **Training Pool Locations:** Excluded from all training sets. Training sets sample strictly from separate cities (`Bhopal`, `Chandigarh`, `Delhi`, `Jaipur`, `Kolkata`, `Mumbai`, etc.).
- **Intersection:** $\emptyset$ (0 locations).

---

## 3. Storage and Freezing
- Benchmark JSON path: `experiments/cognition/cr_exp_15_architecture_binding/benchmark_data/benchmark_1200_sealed.json`
- Frozen SHA-256 recorded in: `CR15_FROZEN_TEST_HASH.txt`
- The file is sealed and read-only for evaluation scripts. No training script or curriculum generator imports or reads this file.

