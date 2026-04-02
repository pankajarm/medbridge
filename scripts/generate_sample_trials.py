"""Generate synthetic clinical trial abstracts for demo purposes."""

import json
from pathlib import Path

SAMPLE_TRIALS = [
    # --- Diabetes (Type 2) ---
    {
        "id": "NCT-DM-001",
        "title": "Cardiovascular Outcomes with Metformin in Type 2 Diabetes",
        "abstract": "This randomized controlled trial evaluated the cardiovascular outcomes of metformin monotherapy in 2,400 patients with type 2 diabetes mellitus over 48 months. Primary endpoint was major adverse cardiovascular events (MACE). Metformin reduced MACE by 18% compared to placebo (HR 0.82, 95% CI 0.71-0.95, p=0.008). Secondary endpoints showed improvement in HbA1c (-1.2%) and fasting glucose. Adverse events included gastrointestinal symptoms in 23% of patients. No significant increase in lactic acidosis was observed.",
        "language": "en", "phase": "III", "country": "US",
        "drugs": ["Metformin"], "diseases": ["Type 2 Diabetes", "Cardiovascular Disease"],
        "adverse_events": [
            {"name": "Gastrointestinal symptoms", "rate": 0.23, "population": "Western"},
            {"name": "Lactic acidosis", "rate": 0.001, "population": "Western"},
        ],
        "enrollment": 2400, "year": 2023,
    },
    {
        "id": "NCT-DM-002",
        "title": "二甲双胍对2型糖尿病心血管结局的影响",
        "abstract": "本随机对照试验评估了二甲双胍单药治疗对1,800名2型糖尿病患者心血管结局的影响，随访36个月。主要终点为主要不良心血管事件(MACE)。二甲双胍组MACE发生率较安慰剂组降低15%(HR 0.85, 95% CI 0.73-0.99, p=0.038)。次要终点显示HbA1c降低1.1%，空腹血糖改善。不良事件包括28%的患者出现胃肠道症状，3%出现维生素B12缺乏。未观察到严重乳酸酸中毒病例。",
        "language": "zh", "phase": "III", "country": "CN",
        "drugs": ["二甲双胍"], "diseases": ["2型糖尿病", "心血管疾病"],
        "adverse_events": [
            {"name": "胃肠道症状", "rate": 0.28, "population": "East Asian"},
            {"name": "维生素B12缺乏", "rate": 0.03, "population": "East Asian"},
        ],
        "enrollment": 1800, "year": 2023,
    },
    {
        "id": "NCT-DM-003",
        "title": "メトホルミンの2型糖尿病における心血管アウトカム試験",
        "abstract": "本無作為化比較試験は、2型糖尿病患者1,200名を対象にメトホルミン単独療法の心血管アウトカムを評価した。追跡期間は36ヶ月。主要評価項目は主要有害心血管イベント（MACE）であった。メトホルミン群はプラセボ群と比較してMACEを14%減少させた（HR 0.86, 95% CI 0.72-1.02, p=0.078）。副次評価項目ではHbA1cが1.0%低下した。有害事象として消化器症状が32%、ビタミンB12欠乏が5%に認められた。日本人集団では消化器症状の発現率が欧米の報告より高かった。",
        "language": "ja", "phase": "III", "country": "JP",
        "drugs": ["メトホルミン"], "diseases": ["2型糖尿病", "心血管疾患"],
        "adverse_events": [
            {"name": "消化器症状", "rate": 0.32, "population": "Japanese"},
            {"name": "ビタミンB12欠乏", "rate": 0.05, "population": "Japanese"},
        ],
        "enrollment": 1200, "year": 2023,
    },
    {
        "id": "NCT-DM-004",
        "title": "Kardiovaskuläre Endpunkte unter Metformin-Therapie bei Typ-2-Diabetes",
        "abstract": "Diese randomisierte kontrollierte Studie untersuchte die kardiovaskulären Endpunkte einer Metformin-Monotherapie bei 1.500 Patienten mit Typ-2-Diabetes mellitus über 42 Monate. Primärer Endpunkt waren schwere kardiovaskuläre Ereignisse (MACE). Metformin reduzierte MACE um 16% im Vergleich zu Placebo (HR 0,84, 95%-KI 0,71-0,99, p=0,041). Sekundäre Endpunkte zeigten eine Verbesserung des HbA1c (-1,1%) und des Nüchternglukosespiegels. Unerwünschte Ereignisse umfassten gastrointestinale Beschwerden bei 21% der Patienten.",
        "language": "de", "phase": "III", "country": "DE",
        "drugs": ["Metformin"], "diseases": ["Typ-2-Diabetes", "Kardiovaskuläre Erkrankung"],
        "adverse_events": [
            {"name": "Gastrointestinale Beschwerden", "rate": 0.21, "population": "Western"},
        ],
        "enrollment": 1500, "year": 2023,
    },
    {
        "id": "NCT-DM-005",
        "title": "Sitagliptin Added to Metformin in Type 2 Diabetes",
        "abstract": "A multicenter trial evaluating sitagliptin 100mg added to metformin in 890 patients with inadequately controlled type 2 diabetes. After 24 weeks, sitagliptin plus metformin reduced HbA1c by an additional 0.7% compared to metformin alone (p<0.001). Weight remained neutral. Adverse events included nasopharyngitis (8%), headache (5%), and upper respiratory tract infection (4%). No cases of pancreatitis were reported. Hypoglycemia was rare (2%).",
        "language": "en", "phase": "III", "country": "US",
        "drugs": ["Sitagliptin", "Metformin"], "diseases": ["Type 2 Diabetes"],
        "adverse_events": [
            {"name": "Nasopharyngitis", "rate": 0.08, "population": "Western"},
            {"name": "Headache", "rate": 0.05, "population": "Western"},
            {"name": "Hypoglycemia", "rate": 0.02, "population": "Western"},
        ],
        "enrollment": 890, "year": 2022,
    },
    {
        "id": "NCT-DM-006",
        "title": "Empagliflozin Cardiovascular Outcome Trial in Type 2 Diabetes",
        "abstract": "This landmark trial enrolled 7,020 patients with type 2 diabetes and established cardiovascular disease. Empagliflozin 10mg/25mg reduced cardiovascular death by 38% (HR 0.62, p<0.001) and heart failure hospitalization by 35%. Overall MACE was reduced by 14%. Notable adverse events included genital mycotic infections (6.4% vs 1.8% placebo) and urinary tract infections (18.0% vs 17.0%). Diabetic ketoacidosis occurred in 0.1% of patients.",
        "language": "en", "phase": "III", "country": "US",
        "drugs": ["Empagliflozin"], "diseases": ["Type 2 Diabetes", "Cardiovascular Disease", "Heart Failure"],
        "adverse_events": [
            {"name": "Genital mycotic infections", "rate": 0.064, "population": "Western"},
            {"name": "Urinary tract infections", "rate": 0.18, "population": "Western"},
            {"name": "Diabetic ketoacidosis", "rate": 0.001, "population": "Western"},
        ],
        "enrollment": 7020, "year": 2023,
    },
    {
        "id": "NCT-DM-007",
        "title": "エンパグリフロジンの日本人2型糖尿病患者における心血管アウトカム",
        "abstract": "日本人2型糖尿病患者1,032名を対象としたエンパグリフロジンの心血管アウトカム試験。エンパグリフロジン10mg群は心血管死亡を32%減少させた（HR 0.68, p=0.012）。心不全入院も28%減少。注目すべき有害事象として、性器真菌感染症が12.1%（プラセボ群2.3%）と欧米の報告（6.4%）より高率であった。尿路感染症は22.0%に認められた。日本人集団では体重減少効果がより顕著であった（-3.2kg vs 欧米-2.5kg）。",
        "language": "ja", "phase": "III", "country": "JP",
        "drugs": ["エンパグリフロジン"], "diseases": ["2型糖尿病", "心血管疾患", "心不全"],
        "adverse_events": [
            {"name": "性器真菌感染症", "rate": 0.121, "population": "Japanese"},
            {"name": "尿路感染症", "rate": 0.22, "population": "Japanese"},
        ],
        "enrollment": 1032, "year": 2023,
    },
    # --- Cardiovascular ---
    {
        "id": "NCT-CV-001",
        "title": "Atorvastatin versus Rosuvastatin in Hypercholesterolemia",
        "abstract": "A head-to-head comparison of atorvastatin 40mg versus rosuvastatin 20mg in 1,650 patients with primary hypercholesterolemia. After 12 weeks, rosuvastatin achieved greater LDL-C reduction (-52.4% vs -46.7%, p<0.001). Both statins similarly reduced cardiovascular events over 3 years. Adverse events: myalgia (atorvastatin 7.2%, rosuvastatin 6.8%), elevated liver enzymes (2.1% vs 1.8%), new-onset diabetes (1.2% vs 1.5%).",
        "language": "en", "phase": "IV", "country": "US",
        "drugs": ["Atorvastatin", "Rosuvastatin"], "diseases": ["Hypercholesterolemia", "Cardiovascular Disease"],
        "adverse_events": [
            {"name": "Myalgia", "rate": 0.072, "population": "Western"},
            {"name": "Elevated liver enzymes", "rate": 0.021, "population": "Western"},
            {"name": "New-onset diabetes", "rate": 0.012, "population": "Western"},
        ],
        "enrollment": 1650, "year": 2022,
    },
    {
        "id": "NCT-CV-002",
        "title": "阿托伐他汀与瑞舒伐他汀在高胆固醇血症中的比较",
        "abstract": "本研究比较了阿托伐他汀40mg与瑞舒伐他汀20mg在1,200名原发性高胆固醇血症患者中的疗效。12周后，瑞舒伐他汀组LDL-C降幅更大（-54.1% vs -48.2%, p<0.001）。不良事件方面，肌痛发生率阿托伐他汀组为9.5%，瑞舒伐他汀组为8.8%。中国人群中肝酶升高发生率（3.8% vs 3.2%）高于西方人群报告。新发糖尿病发生率为1.8% vs 2.1%。",
        "language": "zh", "phase": "IV", "country": "CN",
        "drugs": ["阿托伐他汀", "瑞舒伐他汀"], "diseases": ["高胆固醇血症", "心血管疾病"],
        "adverse_events": [
            {"name": "肌痛", "rate": 0.095, "population": "East Asian"},
            {"name": "肝酶升高", "rate": 0.038, "population": "East Asian"},
            {"name": "新发糖尿病", "rate": 0.018, "population": "East Asian"},
        ],
        "enrollment": 1200, "year": 2022,
    },
    {
        "id": "NCT-CV-003",
        "title": "Aspirina en la prevención primaria cardiovascular en población hispana",
        "abstract": "Ensayo clínico aleatorizado que evaluó aspirina 100mg diaria versus placebo en la prevención primaria cardiovascular en 3,200 pacientes hispanos con factores de riesgo. Seguimiento de 5 años. La aspirina redujo eventos cardiovasculares mayores en un 12% (HR 0.88, IC 95% 0.78-0.99, p=0.034). Sangrado gastrointestinal mayor ocurrió en 1.8% vs 0.9% del grupo placebo. Se observó mayor beneficio en pacientes con diabetes concomitante.",
        "language": "es", "phase": "III", "country": "ES",
        "drugs": ["Aspirina"], "diseases": ["Enfermedad Cardiovascular"],
        "adverse_events": [
            {"name": "Sangrado gastrointestinal", "rate": 0.018, "population": "Hispanic"},
        ],
        "enrollment": 3200, "year": 2023,
    },
    # --- Hypertension ---
    {
        "id": "NCT-HT-001",
        "title": "Amlodipine versus Losartan in Essential Hypertension",
        "abstract": "A randomized trial comparing amlodipine 5-10mg versus losartan 50-100mg in 2,100 patients with essential hypertension. After 24 weeks, both drugs achieved similar blood pressure reduction (amlodipine -15.2/-9.8 mmHg vs losartan -14.8/-9.5 mmHg, p=NS). Adverse events differed significantly: peripheral edema was more common with amlodipine (12.3% vs 2.1%), while cough was more common with losartan (4.5% vs 0.8%). Both were well-tolerated overall.",
        "language": "en", "phase": "III", "country": "US",
        "drugs": ["Amlodipine", "Losartan"], "diseases": ["Essential Hypertension"],
        "adverse_events": [
            {"name": "Peripheral edema", "rate": 0.123, "population": "Western"},
            {"name": "Cough", "rate": 0.008, "population": "Western"},
        ],
        "enrollment": 2100, "year": 2022,
    },
    {
        "id": "NCT-HT-002",
        "title": "Amlodipin versus Losartan bei essentieller Hypertonie",
        "abstract": "Eine randomisierte Studie zum Vergleich von Amlodipin 5-10mg versus Losartan 50-100mg bei 1,800 Patienten mit essentieller Hypertonie. Nach 24 Wochen erreichten beide Medikamente eine vergleichbare Blutdrucksenkung. Periphere Ödeme traten bei Amlodipin häufiger auf (10.8% vs 1.9%). Husten war unter Losartan häufiger (5.2% vs 1.0%). Kopfschmerzen wurden bei 6.3% der Amlodipin-Patienten und 4.8% der Losartan-Patienten berichtet.",
        "language": "de", "phase": "III", "country": "DE",
        "drugs": ["Amlodipin", "Losartan"], "diseases": ["Essentielle Hypertonie"],
        "adverse_events": [
            {"name": "Periphere Ödeme", "rate": 0.108, "population": "Western"},
            {"name": "Husten", "rate": 0.010, "population": "Western"},
            {"name": "Kopfschmerzen", "rate": 0.063, "population": "Western"},
        ],
        "enrollment": 1800, "year": 2022,
    },
    {
        "id": "NCT-HT-003",
        "title": "アムロジピンとロサルタンの本態性高血圧における比較試験",
        "abstract": "本態性高血圧患者950名を対象にアムロジピン5-10mgとロサルタン50-100mgを比較した無作為化試験。24週後の血圧降下は両群で同等であった。有害事象として、アムロジピン群で末梢性浮腫が15.2%（欧米報告12.3%より高率）、ロサルタン群で咳嗽が7.8%に認められた。日本人患者ではアムロジピンによる末梢性浮腫の発現率が欧米より高い傾向が確認された。顔面紅潮はアムロジピン群で8.5%であった。",
        "language": "ja", "phase": "III", "country": "JP",
        "drugs": ["アムロジピン", "ロサルタン"], "diseases": ["本態性高血圧"],
        "adverse_events": [
            {"name": "末梢性浮腫", "rate": 0.152, "population": "Japanese"},
            {"name": "咳嗽", "rate": 0.078, "population": "Japanese"},
            {"name": "顔面紅潮", "rate": 0.085, "population": "Japanese"},
        ],
        "enrollment": 950, "year": 2022,
    },
    {
        "id": "NCT-HT-004",
        "title": "Amlodipine versus Losartan dans l'hypertension artérielle essentielle",
        "abstract": "Essai randomisé comparant l'amlodipine 5-10mg au losartan 50-100mg chez 1,400 patients atteints d'hypertension artérielle essentielle. Après 24 semaines, les deux médicaments ont atteint une réduction tensionnelle comparable. Les œdèmes périphériques étaient plus fréquents sous amlodipine (11.5% vs 2.0%). La toux était plus fréquente sous losartan (4.8% vs 0.9%). Les céphalées ont été rapportées chez 5.9% des patients sous amlodipine.",
        "language": "fr", "phase": "III", "country": "FR",
        "drugs": ["Amlodipine", "Losartan"], "diseases": ["Hypertension artérielle essentielle"],
        "adverse_events": [
            {"name": "Œdèmes périphériques", "rate": 0.115, "population": "Western"},
            {"name": "Toux", "rate": 0.009, "population": "Western"},
        ],
        "enrollment": 1400, "year": 2022,
    },
    # --- Oncology ---
    {
        "id": "NCT-ON-001",
        "title": "Pembrolizumab in Advanced Non-Small Cell Lung Cancer",
        "abstract": "A phase III trial of pembrolizumab 200mg every 3 weeks versus docetaxel in 1,034 patients with previously treated advanced NSCLC expressing PD-L1. Pembrolizumab improved overall survival (median 10.4 vs 8.5 months, HR 0.71, p<0.001). Immune-related adverse events included hypothyroidism (9.1%), pneumonitis (5.8%), and colitis (1.6%). Grade 3-4 treatment-related events were less frequent with pembrolizumab (13.3% vs 35.3%).",
        "language": "en", "phase": "III", "country": "US",
        "drugs": ["Pembrolizumab"], "diseases": ["Non-Small Cell Lung Cancer"],
        "adverse_events": [
            {"name": "Hypothyroidism", "rate": 0.091, "population": "Western"},
            {"name": "Pneumonitis", "rate": 0.058, "population": "Western"},
            {"name": "Colitis", "rate": 0.016, "population": "Western"},
        ],
        "enrollment": 1034, "year": 2023,
    },
    {
        "id": "NCT-ON-002",
        "title": "帕博利珠单抗在晚期非小细胞肺癌中的应用",
        "abstract": "一项III期临床试验，评估帕博利珠单抗200mg每3周一次对比多西他赛治疗既往治疗过的PD-L1阳性晚期非小细胞肺癌（NSCLC）患者800名。帕博利珠单抗组中位总生存期为11.2个月vs多西他赛组8.8个月（HR 0.69, p<0.001）。免疫相关不良事件包括甲状腺功能减退（11.3%，高于西方报告的9.1%）、肺炎（7.2%）和间质性肺病（3.5%，中国人群特有的较高发生率）。3-4级治疗相关事件发生率为15.1%。",
        "language": "zh", "phase": "III", "country": "CN",
        "drugs": ["帕博利珠单抗"], "diseases": ["非小细胞肺癌"],
        "adverse_events": [
            {"name": "甲状腺功能减退", "rate": 0.113, "population": "East Asian"},
            {"name": "肺炎", "rate": 0.072, "population": "East Asian"},
            {"name": "间质性肺病", "rate": 0.035, "population": "East Asian"},
        ],
        "enrollment": 800, "year": 2023,
    },
    {
        "id": "NCT-ON-003",
        "title": "진행성 비소세포폐암에서 펨브롤리주맙의 효과",
        "abstract": "PD-L1 양성 진행성 비소세포폐암(NSCLC) 환자 620명을 대상으로 한 3상 임상시험. 펨브롤리주맙 200mg 3주 간격 투여군과 도세탁셀군을 비교하였다. 펨브롤리주맙군의 중앙 전체 생존기간은 10.8개월 vs 도세탁셀군 8.2개월이었다 (HR 0.72, p=0.001). 면역 관련 이상반응으로 갑상선기능저하증(10.5%), 폐렴(6.5%), 간질성 폐질환(4.2%)이 보고되었다. 한국인 환자에서 간질성 폐질환 발생률이 서양 보고보다 높았다.",
        "language": "ko", "phase": "III", "country": "KR",
        "drugs": ["펨브롤리주맙"], "diseases": ["비소세포폐암"],
        "adverse_events": [
            {"name": "갑상선기능저하증", "rate": 0.105, "population": "Korean"},
            {"name": "폐렴", "rate": 0.065, "population": "Korean"},
            {"name": "간질성 폐질환", "rate": 0.042, "population": "Korean"},
        ],
        "enrollment": 620, "year": 2023,
    },
    # --- Mental Health ---
    {
        "id": "NCT-MH-001",
        "title": "Sertraline versus Escitalopram in Major Depressive Disorder",
        "abstract": "A double-blind trial comparing sertraline 50-200mg to escitalopram 10-20mg in 1,100 patients with major depressive disorder. After 8 weeks, both showed similar efficacy on HAM-D scores (sertraline -12.3 vs escitalopram -12.8, p=0.42). Adverse events: nausea (sertraline 22% vs escitalopram 15%), sexual dysfunction (sertraline 18% vs escitalopram 12%), insomnia (sertraline 14% vs escitalopram 11%), weight gain (sertraline 4% vs escitalopram 6%).",
        "language": "en", "phase": "III", "country": "US",
        "drugs": ["Sertraline", "Escitalopram"], "diseases": ["Major Depressive Disorder"],
        "adverse_events": [
            {"name": "Nausea", "rate": 0.22, "population": "Western"},
            {"name": "Sexual dysfunction", "rate": 0.18, "population": "Western"},
            {"name": "Insomnia", "rate": 0.14, "population": "Western"},
        ],
        "enrollment": 1100, "year": 2022,
    },
    {
        "id": "NCT-MH-002",
        "title": "Sertralin versus Escitalopram bei Major Depression",
        "abstract": "Eine doppelblinde Studie zum Vergleich von Sertralin 50-200mg mit Escitalopram 10-20mg bei 850 Patienten mit Major Depression. Nach 8 Wochen zeigten beide eine vergleichbare Wirksamkeit auf der HAM-D-Skala. Unerwünschte Ereignisse: Übelkeit (Sertralin 20% vs Escitalopram 14%), sexuelle Dysfunktion (Sertralin 16% vs Escitalopram 10%), Schlaflosigkeit (Sertralin 12% vs Escitalopram 9%). Die Gewichtszunahme war unter Escitalopram etwas häufiger (7% vs 5%).",
        "language": "de", "phase": "III", "country": "DE",
        "drugs": ["Sertralin", "Escitalopram"], "diseases": ["Major Depression"],
        "adverse_events": [
            {"name": "Übelkeit", "rate": 0.20, "population": "Western"},
            {"name": "Sexuelle Dysfunktion", "rate": 0.16, "population": "Western"},
            {"name": "Schlaflosigkeit", "rate": 0.12, "population": "Western"},
        ],
        "enrollment": 850, "year": 2022,
    },
    {
        "id": "NCT-MH-003",
        "title": "セルトラリンとエスシタロプラムの大うつ病性障害における比較",
        "abstract": "大うつ病性障害患者680名を対象にセルトラリン50-200mgとエスシタロプラム10-20mgを比較した二重盲検試験。8週後のHAM-Dスコアは両群で同等であった。有害事象として、悪心（セルトラリン28% vs エスシタロプラム18%）、性機能障害（セルトラリン8% vs エスシタロプラム5%）が報告された。日本人患者では悪心の発現率が欧米より高く、性機能障害の報告率は低かった。体重増加はエスシタロプラム群で多かった（9% vs 4%）。",
        "language": "ja", "phase": "III", "country": "JP",
        "drugs": ["セルトラリン", "エスシタロプラム"], "diseases": ["大うつ病性障害"],
        "adverse_events": [
            {"name": "悪心", "rate": 0.28, "population": "Japanese"},
            {"name": "性機能障害", "rate": 0.08, "population": "Japanese"},
            {"name": "体重増加", "rate": 0.04, "population": "Japanese"},
        ],
        "enrollment": 680, "year": 2022,
    },
]

# Cross-lingual drug name mappings for entity resolution
DRUG_ALIASES = {
    "Metformin": ["二甲双胍", "メトホルミン"],
    "Empagliflozin": ["エンパグリフロジン"],
    "Sitagliptin": ["シタグリプチン", "西格列汀"],
    "Atorvastatin": ["阿托伐他汀", "アトルバスタチン"],
    "Rosuvastatin": ["瑞舒伐他汀", "ロスバスタチン"],
    "Aspirin": ["Aspirina", "アスピリン"],
    "Amlodipine": ["Amlodipin", "アムロジピン", "Amlodipine"],
    "Losartan": ["ロサルタン", "Losartan"],
    "Pembrolizumab": ["帕博利珠单抗", "펨브롤리주맙", "ペムブロリズマブ"],
    "Sertraline": ["Sertralin", "セルトラリン"],
    "Escitalopram": ["エスシタロプラム"],
}

# Known drug interactions
DRUG_INTERACTIONS = [
    {"drug1": "Metformin", "drug2": "Empagliflozin", "type": "synergistic", "severity": "low"},
    {"drug1": "Metformin", "drug2": "Sitagliptin", "type": "synergistic", "severity": "low"},
    {"drug1": "Atorvastatin", "drug2": "Amlodipine", "type": "pharmacokinetic", "severity": "moderate"},
    {"drug1": "Rosuvastatin", "drug2": "Aspirin", "type": "additive", "severity": "low"},
    {"drug1": "Sertraline", "drug2": "Aspirin", "type": "bleeding_risk", "severity": "moderate"},
    {"drug1": "Losartan", "drug2": "Aspirin", "type": "reduced_efficacy", "severity": "moderate"},
]


def main():
    output_dir = Path(__file__).parent.parent / "data" / "sample_trials"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save individual trial files
    for trial in SAMPLE_TRIALS:
        filepath = output_dir / f"{trial['id']}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(trial, f, ensure_ascii=False, indent=2)

    # Save drug aliases
    aliases_path = output_dir / "_drug_aliases.json"
    with open(aliases_path, "w", encoding="utf-8") as f:
        json.dump(DRUG_ALIASES, f, ensure_ascii=False, indent=2)

    # Save drug interactions
    interactions_path = output_dir / "_drug_interactions.json"
    with open(interactions_path, "w", encoding="utf-8") as f:
        json.dump(DRUG_INTERACTIONS, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(SAMPLE_TRIALS)} sample trials in {output_dir}")
    print(f"Drug aliases: {len(DRUG_ALIASES)} mappings")
    print(f"Drug interactions: {len(DRUG_INTERACTIONS)} pairs")


if __name__ == "__main__":
    main()
