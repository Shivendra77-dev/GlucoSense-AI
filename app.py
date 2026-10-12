"""
GlucoSense AI - Streamlit app
Shows the mobile UI design and runs the real logistic-regression model (model.json) in the page.
Run:  streamlit run app.py
"""
import json
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="GlucoSense AI", page_icon="🩺", layout="centered")

st.markdown(
    """
<style>
header[data-testid="stHeader"], footer, #MainMenu {display:none !important;}
.stApp {background:#faf8ff;}
.block-container {max-width:470px; padding:0 !important;}
iframe {border:0; display:block;}
</style>""",
    unsafe_allow_html=True,
)

M = json.loads((Path(__file__).parent / "model.json").read_text())
MODEL_JS = json.dumps({k: M[k] for k in ("median", "mean", "scale", "coef", "intercept")})

HTML = r"""<html class="light" lang="en"><head><meta charset="utf-8"/><meta content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover" name="viewport"/><link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&amp;display=swap" rel="stylesheet"/><link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" rel="stylesheet"/>
<style>@layer base{html,body{width:100%;margin:0;padding:0;background-color:#faf8ff;}body{overscroll-behavior:none;-webkit-tap-highlight-color:transparent;font-family:Inter,sans-serif;}.pb-safe{padding-bottom:env(safe-area-inset-bottom,0px);}.pt-safe{padding-top:env(safe-area-inset-top,0px);}main>:first-child{margin-top:0!important;}main>:last-child{margin-bottom:0!important;}}::-webkit-scrollbar{display:none;}</style><script src="https://cdn.tailwindcss.com"></script><script id="tailwind-config">tailwind.config = { darkMode: "class", theme: { extend: { colors: { "on-secondary": "#ffffff", "primary-fixed": "#89f5e7", "tertiary": "#ba0035", "surface-bright": "#faf8ff", "on-primary": "#ffffff", "on-surface-variant": "#3d4947", "on-tertiary-fixed": "#40000c", "on-tertiary-fixed-variant": "#920028", "on-secondary-fixed-variant": "#005137", "surface": "#faf8ff", "outline": "#6d7a77", "primary-container": "#008378", "tertiary-container": "#e21e49", "inverse-surface": "#283044", "secondary-container": "#82f5c1", "secondary-fixed-dim": "#68dba9", "primary-fixed-dim": "#6bd8cb", "on-primary-fixed-variant": "#005049", "surface-container-low": "#f2f3ff", "surface-container-highest": "#dae2fd", "on-secondary-fixed": "#002114", "on-primary-container": "#f4fffc", "surface-container-lowest": "#ffffff", "error": "#ba1a1a", "surface-dim": "#d2d9f4", "on-background": "#131b2e", "surface-container-high": "#e2e7ff", "tertiary-fixed-dim": "#ffb3b6", "surface-tint": "#006a61", "tertiary-fixed": "#ffdada", "outline-variant": "#bcc9c6", "surface-container": "#eaedff", "on-error": "#ffffff", "on-tertiary": "#ffffff", "primary": "#00685f", "on-primary-fixed": "#00201d", "on-secondary-container": "#00714e", "inverse-primary": "#6bd8cb", "background": "#faf8ff", "on-tertiary-container": "#fffbff", "secondary-fixed": "#85f8c4", "error-container": "#ffdad6", "surface-variant": "#dae2fd", "on-surface": "#131b2e", "secondary": "#006c4a", "inverse-on-surface": "#eef0ff", "on-error-container": "#93000a" }, borderRadius: { "DEFAULT": "0.25rem", "lg": "0.5rem", "xl": "0.75rem", "full": "9999px" }, spacing: { "space-sm": "0.5rem", "space-md": "1rem", "space-xs": "0.25rem", "space-lg": "1.5rem", "margin-desktop": "2.5rem", "margin": "1rem", "gutter-desktop": "1.5rem", "gutter": "1.25rem", "space-xl": "2rem", "margin-tablet": "1.5rem" }, fontFamily: { "display-lg-mobile": ["Inter"], "body-lg": ["Inter"], "display-lg": ["Inter"], "metric-numeral": ["Inter"], "title-md": ["Inter"], "headline-lg-mobile": ["Inter"], "label-lg": ["Inter"], "headline-md": ["Inter"], "label-sm": ["Inter"], "body-md": ["Inter"], "headline-sm": ["Inter"], "body-sm": ["Inter"], "label-md": ["Inter"], "headline-lg": ["Inter"] }, fontSize: { "display-lg-mobile": ["36px", { lineHeight: "44px", letterSpacing: "-0.02em", fontWeight: "700" }], "body-lg": ["16px", { lineHeight: "26px", letterSpacing: "0em", fontWeight: "400" }], "display-lg": ["48px", { lineHeight: "56px", letterSpacing: "-0.025em", fontWeight: "700" }], "metric-numeral": ["28px", { lineHeight: "32px", letterSpacing: "-0.03em", fontWeight: "700" }], "title-md": ["16px", { lineHeight: "24px", letterSpacing: "-0.005em", fontWeight: "600" }], "headline-lg-mobile": ["26px", { lineHeight: "34px", letterSpacing: "-0.015em", fontWeight: "600" }], "label-lg": ["14px", { lineHeight: "20px", letterSpacing: "0.01em", fontWeight: "600" }], "headline-md": ["24px", { lineHeight: "32px", letterSpacing: "-0.015em", fontWeight: "600" }], "label-sm": ["11px", { lineHeight: "14px", letterSpacing: "0.04em", fontWeight: "600" }], "body-md": ["14px", { lineHeight: "22px", letterSpacing: "0em", fontWeight: "400" }], "headline-sm": ["20px", { lineHeight: "28px", letterSpacing: "-0.01em", fontWeight: "600" }], "body-sm": ["12px", { lineHeight: "18px", letterSpacing: "0.005em", fontWeight: "400" }], "label-md": ["12px", { lineHeight: "16px", letterSpacing: "0.02em", fontWeight: "500" }], "headline-lg": ["32px", { lineHeight: "40px", letterSpacing: "-0.02em", fontWeight: "600" }] } } } };</script></head><body class="bg-surface text-on-surface font-body-md text-body-md flex flex-col min-h-screen selection:bg-primary-fixed selection:text-on-primary-fixed"><header class="fixed top-0 w-full z-50 bg-surface/80 backdrop-blur-xl shadow-[0_1px_8px_rgba(0,0,0,0.04)] pt-safe"><div class="h-16 px-margin flex items-center justify-between gap-space-sm"><div class="flex items-center gap-space-sm min-w-0"><img alt="GlucoSense AI Logo" class="h-8 w-auto object-contain shrink-0" src="https://lh3.googleusercontent.com/aida-public/AB6AXuAcKFIq6FhdnempHvdyuUvKXifi6gbGdd638v27z5DhrLgXXlNQKBYm6rQCw_M9X2Sz9_PUGCociSuPkLF7bTm-if7ipzAFFT0i0vVcualHg_nYk8Oh5pklaeAWQjs_YsNLAnDmtoRng8eMS7LQkqexNJnZ19ATueTaKo4Q6gE3sxKCVlrwNI7q8rcARN8gXwx6b-ilQfy3t8pLRaz7uKNvZ1pSKXWLh_mJb6apKI2z6FMHMHHAo7vZYQ"/><div class="flex flex-col min-w-0"><div class="flex items-center gap-space-xs"><span class="font-title-md text-title-md text-on-surface truncate">GlucoSense AI</span><span class="hidden sm:inline-block font-label-sm text-label-sm px-1.5 py-0.5 rounded-full bg-secondary-container text-on-secondary-container font-semibold">Live</span></div><span class="font-label-sm text-label-sm text-on-surface-variant truncate">Patient Assessment</span></div></div><div class="flex items-center gap-space-xs shrink-0"><div class="flex items-center gap-1.5 px-2 py-1 rounded-full bg-surface-container-low"><span class="w-2 h-2 rounded-full bg-secondary animate-pulse"></span><span class="font-label-sm text-label-sm text-secondary font-medium hidden xs:inline">Model online</span></div><button aria-label="Doctor Profile" class="w-11 h-11 rounded-full flex items-center justify-center p-0.5 hover:bg-surface-container-high transition-colors" type="button"><img alt="Profile" class="w-8 h-8 rounded-full object-cover" src="https://lh3.googleusercontent.com/aida-public/AB6AXuCZAEboUkMItRdpgBUN1V15sLHYZiSOXN8hdjChmIGWN46RFiF_0m4ZmY_HXHMiNYQ1HjXmU1ijp758K-KjetCPA0p6RPjvugD5o56GAUrP-iU22U0Rp8lCoOgZ89XTN1x2GO7eUdvjLJ-0E4l9KUqsBSRM7VsKeBEkN7zHdgvc6ILVMLSgx5W7s9fztgsm1YhpK8rYbTjaX40w0WuBtd--X7VEwdDW-29o-hn8xjd_MTJg_Yg3dE0sEQ"/></button></div></div></header><main class="flex flex-col relative w-full pt-16 pb-24 bg-surface min-h-screen"><div class="flex flex-col w-full">
<div class="px-margin pt-space-md pb-space-sm flex flex-col gap-space-xs">
<div class="flex items-center justify-between">
<div class="flex items-center gap-space-xs">
<span class="inline-flex items-center justify-center w-6 h-6 rounded-full bg-primary text-on-primary font-label-sm text-label-sm">
<span class="material-symbols-outlined text-[14px]">vital_signs</span>
</span>
<span class="font-label-sm text-label-sm uppercase tracking-wider text-primary font-semibold">Diagnostic Intake</span>
</div>
<div class="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-secondary-container/40">
<span class="w-2 h-2 rounded-full bg-secondary animate-ping"></span>
<span class="font-label-sm text-label-sm text-on-secondary-container font-semibold">8 Parameters</span>
</div>
</div>
<h1 class="font-headline-lg-mobile text-headline-lg-mobile text-on-surface">Diabetes Risk Assessment</h1>
<p class="font-body-sm text-body-sm text-on-surface-variant">8-feature screening model trained on 768 patient records from the Pima Indians Diabetes dataset.</p>
</div>
<div class="px-margin my-space-xs">
<div class="p-space-md rounded-xl bg-surface-container-low shadow-sm flex flex-col gap-space-sm">
<div class="flex items-center justify-between">
<div class="flex items-center gap-space-xs">
<span class="material-symbols-outlined text-primary text-[20px]">verified_user</span>
<span class="font-label-md text-label-md text-on-surface font-semibold">Standardized Patient Intake</span>
</div>
<span class="font-label-sm text-label-sm px-2 py-0.5 rounded-full bg-surface-container-high text-on-surface-variant font-semibold" id="completenessBadge">8/8 Ready</span>
</div>
<div class="w-full bg-surface-container-high h-2 rounded-full overflow-hidden">
<div class="bg-primary h-full rounded-full transition-all duration-300" id="progressFill" style="width: 100%;"></div>
</div>
</div>
</div>
<div class="px-margin py-space-xs">
<div class="flex items-center gap-1.5 p-1 rounded-xl bg-surface-container-low" role="tablist">
<button class="flex-1 py-2 px-2.5 rounded-lg font-label-md text-label-md text-center transition-all bg-surface-container-lowest text-primary shadow-sm font-semibold flex items-center justify-center gap-1" id="tabDemographics" onclick="switchSection('demographics')" role="tab" type="button">
<span class="material-symbols-outlined text-[16px]">person</span>
<span>Biometrics</span>
</button>
<button class="flex-1 py-2 px-2.5 rounded-lg font-label-md text-label-md text-center transition-all text-on-surface-variant hover:text-on-surface flex items-center justify-center gap-1" id="tabVitals" onclick="switchSection('vitals')" role="tab" type="button">
<span class="material-symbols-outlined text-[16px]">monitor_heart</span>
<span>Lab Vitals</span>
</button>
<button class="flex-1 py-2 px-2.5 rounded-lg font-label-md text-label-md text-center transition-all text-on-surface-variant hover:text-on-surface flex items-center justify-center gap-1" id="tabGenetics" onclick="switchSection('genetics')" role="tab" type="button">
<span class="material-symbols-outlined text-[16px]">strikethrough_s</span>
<span>Genetics</span>
</button>
</div>
</div>
<form class="px-margin flex flex-col gap-space-md mt-space-sm" id="assessmentForm" onsubmit="event.preventDefault(); triggerAssessment();">
<section class="flex flex-col gap-space-md" id="sec-demographics">
<div class="p-space-lg rounded-xl bg-surface-container-lowest shadow-sm flex flex-col gap-space-lg">
<div class="flex items-center justify-between">
<div class="flex items-center gap-space-sm">
<span class="w-8 h-8 rounded-lg bg-surface-container-high flex items-center justify-center text-primary">
<span class="material-symbols-outlined text-[20px]">badge</span>
</span>
<div>
<h2 class="font-title-md text-title-md text-on-surface">Demographics &amp; Biometrics</h2>
<p class="font-body-sm text-body-sm text-on-surface-variant">Baseline physical characteristics</p>
</div>
</div>
<span class="font-label-sm text-label-sm px-2 py-0.5 rounded-full bg-surface-container text-on-surface font-semibold">Part 1/3</span>
</div>
<div class="p-space-md rounded-xl bg-surface-container-low flex flex-col gap-space-sm">
<div class="flex items-center justify-between">
<label class="font-label-lg text-label-lg text-on-surface flex items-center gap-1" for="ageSlider">
<span>Patient Age</span>
<span class="font-body-sm text-body-sm text-on-surface-variant">(years)</span>
</label>
<div class="flex items-center gap-1.5">
<button aria-label="Decrease age" class="w-8 h-8 rounded-lg bg-surface-container-lowest text-on-surface hover:bg-surface-container-high active:scale-95 transition-all flex items-center justify-center shadow-sm" onclick="adjustValue('ageInput', -1)" type="button">
<span class="material-symbols-outlined text-[18px]">remove</span>
</button>
<input class="w-14 h-8 text-center font-metric-numeral text-[18px] text-primary bg-surface-container-lowest rounded-lg shadow-sm focus:outline-none" id="ageInput" max="81" min="21" oninput="syncAgeFromNum(this.value)" type="number" value="33"/>
<button aria-label="Increase age" class="w-8 h-8 rounded-lg bg-surface-container-lowest text-on-surface hover:bg-surface-container-high active:scale-95 transition-all flex items-center justify-center shadow-sm" onclick="adjustValue('ageInput', 1)" type="button">
<span class="material-symbols-outlined text-[18px]">add</span>
</button>
</div>
</div>
<input class="w-full accent-primary h-2 bg-surface-container-highest rounded-lg cursor-pointer" id="ageSlider" max="81" min="21" oninput="syncAgeFromSlider(this.value)" type="range" value="33"/>
<div class="flex justify-between font-label-sm text-label-sm text-on-surface-variant px-0.5">
<span>21 yr min</span>
<span class="text-primary font-semibold" id="ageCohortLabel">Adult (33 yr)</span>
<span>81 yr max</span>
</div>
</div>
<div class="p-space-md rounded-xl bg-surface-container-low flex flex-col gap-space-sm">
<div class="flex items-start justify-between gap-space-sm">
<div>
<div class="flex items-center gap-1.5">
<label class="font-label-lg text-label-lg text-on-surface" for="bmiInput">Body Mass Index (BMI)</label>
<button aria-label="BMI formula info" class="text-on-surface-variant hover:text-primary transition-colors" onclick="toggleBmiInfo()" type="button">
<span class="material-symbols-outlined text-[18px]">info</span>
</button>
</div>
<p class="font-body-sm text-body-sm text-on-surface-variant">Weight in kg / (height in m)²</p>
</div>
<div class="flex flex-col items-end">
<span class="font-label-sm text-label-sm px-2.5 py-0.5 rounded-full bg-secondary-container text-on-secondary-container font-semibold" id="bmiClassBadge">Normative</span>
<span class="font-body-sm text-body-sm text-on-surface-variant mt-0.5">kg/m²</span>
</div>
</div>
<div class="hidden p-space-sm rounded-lg bg-surface-container-highest text-on-surface font-body-sm text-body-sm flex items-start gap-space-xs" id="bmiInfoCard">
<span class="material-symbols-outlined text-primary text-[18px] shrink-0 mt-0.5">calculate</span>
<div>Standard WHO threshold calculation: Underweight (&lt;18.5), Normal (18.5–24.9), Overweight (25–29.9), Obese (≥30).</div>
</div>
<div class="flex items-center gap-space-sm">
<input class="flex-1 h-11 px-space-md rounded-lg bg-surface-container-lowest text-on-surface font-title-md text-title-md shadow-sm focus:outline-none" id="bmiInput" max="55.0" min="18.0" oninput="updateBmiState(this.value)" step="0.1" type="number" value="26.4"/>
<div class="flex gap-1">
<button class="px-2.5 py-2 rounded-lg bg-surface-container-lowest text-on-surface font-label-md text-label-md shadow-sm hover:bg-surface-container-high" onclick="setBmiPreset(22.4)" type="button">22.4</button>
<button class="px-2.5 py-2 rounded-lg bg-surface-container-lowest text-on-surface font-label-md text-label-md shadow-sm hover:bg-surface-container-high" onclick="setBmiPreset(28.2)" type="button">28.2</button>
<button class="px-2.5 py-2 rounded-lg bg-surface-container-lowest text-on-surface font-label-md text-label-md shadow-sm hover:bg-surface-container-high" onclick="setBmiPreset(34.8)" type="button">34.8</button>
</div>
</div>
<div class="w-full bg-surface-container-highest h-1.5 rounded-full overflow-hidden mt-1">
<div class="h-full bg-secondary transition-all" id="bmiMeterBar" style="width: 42%;"></div>
</div>
</div>
</div>
</section>
<section class="flex flex-col gap-space-md" id="sec-vitals">
<div class="p-space-lg rounded-xl bg-surface-container-lowest shadow-sm flex flex-col gap-space-md">
<div class="flex items-center justify-between">
<div class="flex items-center gap-space-sm">
<span class="w-8 h-8 rounded-lg bg-surface-container-high flex items-center justify-center text-primary">
<span class="material-symbols-outlined text-[20px]">science</span>
</span>
<div>
<h2 class="font-title-md text-title-md text-on-surface">Clinical Lab Vitals</h2>
<p class="font-body-sm text-body-sm text-on-surface-variant">Serum markers &amp; hemodynamic data</p>
</div>
</div>
<span class="font-label-sm text-label-sm px-2 py-0.5 rounded-full bg-surface-container text-on-surface font-semibold">Part 2/3</span>
</div>
<div class="p-space-md rounded-xl bg-surface-container-low flex flex-col gap-space-xs">
<div class="flex items-center justify-between">
<div class="flex flex-col">
<span class="font-label-lg text-label-lg text-on-surface">Pregnancies</span>
<span class="font-body-sm text-body-sm text-on-surface-variant">Number of times pregnant</span>
</div>
<div class="flex items-center gap-2">
<button aria-label="Decrease pregnancies" class="w-8 h-8 rounded-lg bg-surface-container-lowest text-on-surface hover:bg-surface-container-high active:scale-95 flex items-center justify-center shadow-sm" onclick="adjustValue('pregInput', -1)" type="button">
<span class="material-symbols-outlined text-[18px]">remove</span>
</button>
<input class="w-12 h-8 text-center font-metric-numeral text-[18px] text-on-surface bg-surface-container-lowest rounded-lg shadow-sm focus:outline-none" id="pregInput" max="17" min="0" type="number" value="1"/>
<button aria-label="Increase pregnancies" class="w-8 h-8 rounded-lg bg-surface-container-lowest text-on-surface hover:bg-surface-container-high active:scale-95 flex items-center justify-center shadow-sm" onclick="adjustValue('pregInput', 1)" type="button">
<span class="material-symbols-outlined text-[18px]">add</span>
</button>
</div>
</div>
</div>
<div class="p-space-md rounded-xl bg-surface-container-low flex flex-col gap-space-sm">
<div class="flex items-center justify-between">
<div>
<div class="flex items-center gap-1.5">
<span class="font-label-lg text-label-lg text-on-surface">2-Hour Plasma Glucose</span>
<span class="font-label-sm text-label-sm px-2 py-0.5 rounded-full bg-secondary-container text-on-secondary-container font-semibold">Healthy &lt;140</span>
</div>
<p class="font-body-sm text-body-sm text-on-surface-variant">2 hours after a glucose tolerance test (mg/dL)</p>
</div>
<span class="font-metric-numeral text-metric-numeral text-primary" id="glucoseValueDisplay">115</span>
</div>
<input class="w-full accent-primary h-2 bg-surface-container-highest rounded-lg cursor-pointer" id="glucoseSlider" max="250" min="50" oninput="updateGlucoseDisplay(this.value)" type="range" value="115"/>
<div class="flex justify-between font-label-sm text-label-sm text-on-surface-variant">
<span>50 mg/dL</span>
<span class="text-secondary font-semibold" id="glucoseStatusTag">Normal (115)</span>
<span>250 mg/dL</span>
</div>
</div>
<div class="grid grid-cols-1 sm:grid-cols-2 gap-space-sm">
<div class="p-space-md rounded-xl bg-surface-container-low flex flex-col gap-space-xs">
<div class="flex items-center justify-between">
<label class="font-label-md text-label-md text-on-surface" for="bpInput">Diastolic BP</label>
<span class="font-label-sm text-label-sm text-on-surface-variant">mm Hg</span>
</div>
<div class="flex items-center gap-2">
<input class="w-full h-10 px-3 rounded-lg bg-surface-container-lowest text-on-surface font-title-md text-title-md shadow-sm focus:outline-none" id="bpInput" max="130" min="50" type="number" value="76"/>
<span class="font-label-sm text-label-sm px-2 py-1 rounded bg-secondary-container text-on-secondary-container">Optimal</span>
</div>
<span class="font-body-sm text-body-sm text-on-surface-variant">Standard cuff baseline: 60-85</span>
</div>
<div class="p-space-md rounded-xl bg-surface-container-low flex flex-col gap-space-xs">
<div class="flex items-center justify-between">
<label class="font-label-md text-label-md text-on-surface" for="skinInput">Triceps Skinfold</label>
<span class="font-label-sm text-label-sm text-on-surface-variant">mm</span>
</div>
<div class="flex items-center gap-2">
<input class="w-full h-10 px-3 rounded-lg bg-surface-container-lowest text-on-surface font-title-md text-title-md shadow-sm focus:outline-none" id="skinInput" max="60" min="10" type="number" value="25"/>
<span class="font-label-sm text-label-sm px-2 py-1 rounded bg-surface-container-high text-on-surface">Subcut</span>
</div>
<span class="font-body-sm text-body-sm text-on-surface-variant">Caliper subcutaneous fat estimate</span>
</div>
</div>
<div class="p-space-md rounded-xl bg-surface-container-low flex flex-col gap-space-xs">
<div class="flex items-center justify-between">
<div>
<span class="font-label-lg text-label-lg text-on-surface">2-Hour Serum Insulin</span>
<p class="font-body-sm text-body-sm text-on-surface-variant">Postprandial glycemic reaction (μU/mL)</p>
</div>
<span class="font-metric-numeral text-[22px] text-primary" id="insulinValText">85</span>
</div>
<div class="flex items-center gap-space-sm mt-1">
<input class="w-full accent-primary h-2 bg-surface-container-highest rounded-lg cursor-pointer" id="insulinSlider" max="450" min="15" oninput="updateInsulinDisplay(this.value)" type="range" value="85"/>
<input class="w-16 h-8 text-center bg-surface-container-lowest rounded-lg font-label-md text-label-md shadow-sm focus:outline-none" id="insulinNum" max="450" min="15" oninput="syncInsulinSlider(this.value)" type="number" value="85"/>
</div>
<div class="flex justify-between font-label-sm text-label-sm text-on-surface-variant">
<span>15 (Fasting base)</span>
<span class="text-secondary font-semibold">Standard Responders: 16-166</span>
<span>450 μU/mL</span>
</div>
</div>
</div>
</section>
<section class="flex flex-col gap-space-md" id="sec-genetics">
<div class="p-space-lg rounded-xl bg-surface-container-lowest shadow-sm flex flex-col gap-space-md">
<div class="flex items-center justify-between">
<div class="flex items-center gap-space-sm">
<span class="w-8 h-8 rounded-lg bg-surface-container-high flex items-center justify-center text-primary">
<span class="material-symbols-outlined text-[20px]">family_restroom</span>
</span>
<div>
<h2 class="font-title-md text-title-md text-on-surface">Genetic Predisposition</h2>
<p class="font-body-sm text-body-sm text-on-surface-variant">Pedigree lineage assessment index</p>
</div>
</div>
<span class="font-label-sm text-label-sm px-2 py-0.5 rounded-full bg-surface-container text-on-surface font-semibold">Part 3/3</span>
</div>
<div class="p-space-md rounded-xl bg-surface-container-low flex flex-col gap-space-sm">
<div class="flex items-start justify-between">
<div>
<div class="flex items-center gap-1.5">
<span class="font-label-lg text-label-lg text-on-surface">Diabetes Pedigree Function</span>
<button aria-label="Pedigree function explanation" class="text-on-surface-variant hover:text-primary transition-colors" onclick="togglePedigreeInfo()" type="button">
<span class="material-symbols-outlined text-[18px]">help</span>
</button>
</div>
<p class="font-body-sm text-body-sm text-on-surface-variant">Quantifies hereditary risk based on pedigree relatives</p>
</div>
<span class="font-metric-numeral text-metric-numeral text-primary" id="dpfScore">0.47</span>
</div>
<div class="hidden p-space-sm rounded-lg bg-surface-container-highest text-on-surface font-body-sm text-body-sm flex flex-col gap-1" id="dpfInfoCard">
<div class="flex items-center gap-1 font-label-md text-label-md text-primary font-semibold">
<span class="material-symbols-outlined text-[16px]">account_tree</span>
<span>Heritability Synthesis Formula</span>
</div>
<p>Scores compute cumulative family history, genetic distance to relatives with diabetes, and age of familial symptom onset. High values (&gt;1.0) reflect dense immediate kinship incidence.</p>
</div>
<input class="w-full accent-primary h-2 bg-surface-container-highest rounded-lg cursor-pointer" id="dpfSlider" max="2.50" min="0.05" oninput="updateDpfDisplay(this.value)" step="0.01" type="range" value="0.47"/>
<div class="grid grid-cols-3 gap-2 text-center pt-1">
<button class="p-2 rounded-lg bg-surface-container-lowest text-on-surface hover:bg-surface-container-high transition-colors flex flex-col items-center" onclick="setDpf(0.24)" type="button">
<span class="font-label-sm text-label-sm text-on-surface-variant">Low Heredity</span>
<span class="font-label-md text-label-md font-semibold text-secondary">0.24</span>
</button>
<button class="p-2 rounded-lg bg-surface-container-lowest text-on-surface ring-2 ring-primary hover:bg-surface-container-high transition-colors flex flex-col items-center" onclick="setDpf(0.47)" type="button">
<span class="font-label-sm text-label-sm text-on-surface-variant">Population Avg</span>
<span class="font-label-md text-label-md font-semibold text-primary">0.47</span>
</button>
<button class="p-2 rounded-lg bg-surface-container-lowest text-on-surface hover:bg-surface-container-high transition-colors flex flex-col items-center" onclick="setDpf(1.35)" type="button">
<span class="font-label-sm text-label-sm text-on-surface-variant">Strong History</span>
<span class="font-label-md text-label-md font-semibold text-tertiary">1.35</span>
</button>
</div>
</div>
</div>
</section>
<div class="p-space-md rounded-xl bg-tertiary-fixed/30 flex items-start gap-space-sm shadow-sm">
<span class="material-symbols-outlined text-tertiary text-[20px] shrink-0 mt-0.5">warning</span>
<div class="flex flex-col gap-0.5">
<span class="font-label-md text-label-md text-on-tertiary-fixed font-semibold">Data Integrity Guideline</span>
<p class="font-body-sm text-body-sm text-on-surface-variant">
          Missing physiological measurements should not be left as 0; clinical imputation guidelines apply. Values are strictly checked against plausible human metabolic limits.
        </p>
</div>
</div>
<div class="flex flex-col gap-space-sm mt-space-xs">
<button class="relative overflow-hidden w-full h-14 rounded-xl bg-primary text-on-primary font-label-lg text-label-lg flex items-center justify-center gap-2 shadow-lg transition-all active:scale-[0.99] group" id="runAssessmentBtn" type="submit">
<span class="absolute inset-0 bg-primary-fixed/20 opacity-0 group-hover:opacity-100 transition-opacity"></span>
<span class="relative flex items-center gap-2">
<span class="material-symbols-outlined text-[22px] animate-pulse">neurology</span>
<span class="font-semibold tracking-wide">Run Assessment</span>
<span class="material-symbols-outlined text-[18px]">arrow_forward</span>
</span>
</button>
<button class="w-full h-11 rounded-xl bg-surface-container text-on-surface font-label-md text-label-md flex items-center justify-center gap-2 hover:bg-surface-container-high transition-colors" onclick="resetAssessmentForm()" type="button">
<span class="material-symbols-outlined text-[18px]">restart_alt</span>
<span>Reset Form Values</span>
</button>
</div>
<div class="hidden p-space-lg rounded-xl bg-surface-container-lowest shadow-xl flex flex-col gap-space-md" id="resultModal">
<div class="flex items-center justify-between">
<div class="flex items-center gap-2">
<span class="w-3 h-3 rounded-full bg-secondary"></span>
<span class="font-title-md text-title-md text-on-surface font-semibold">Inference Ready</span>
</div>
<button class="text-on-surface-variant hover:text-on-surface" onclick="closeResultModal()" type="button">
<span class="material-symbols-outlined text-[20px]">close</span>
</button>
</div>
<div class="p-space-md rounded-xl bg-surface-container-low flex items-center justify-between">
<div>
<span class="font-label-sm text-label-sm text-on-surface-variant uppercase">Prediction Score</span>
<div class="font-display-lg-mobile text-display-lg-mobile text-primary font-bold" id="resPct">0%</div>
<span class="font-label-sm text-label-sm text-secondary font-semibold" id="resLabel">Lower Metabolic Risk</span>
</div>
<div class="w-16 h-16 rounded-full bg-secondary-container flex items-center justify-center" id="resIconWrap">
<span class="material-symbols-outlined text-on-secondary-container text-[32px]" id="resIcon">check_circle</span>
</div>
</div>
<p class="font-body-sm text-body-sm text-on-surface-variant" id="resText"></p>
<div class="flex flex-col gap-1.5" id="resBars"></div>
<div class="flex gap-2">
<button class="flex-1 h-10 rounded-lg bg-primary text-on-primary font-label-md text-label-md flex items-center justify-center gap-1 shadow-sm font-semibold" onclick="document.getElementById('resBars').scrollIntoView({behavior:'smooth',block:'center'})" type="button">
<span>View Deep Insights</span>
<span class="material-symbols-outlined text-[16px]">trending_up</span>
</button>
<button class="px-4 h-10 rounded-lg bg-surface-container text-on-surface font-label-md text-label-md font-semibold" onclick="closeResultModal()" type="button">Dismiss</button>
</div>
</div>
</form>
</div>
<script>
  const MODEL = __MODEL__;

  /*PREDICT_START*/
  const FEATURES = ['Pregnancies', 'Glucose', 'Blood pressure', 'Skin fold', 'Insulin', 'BMI', 'Family history', 'Age'];
  const RANGES = [[0, 17], [50, 250], [40, 130], [5, 60], [10, 600], [15, 60], [0.05, 2.5], [18, 90]];
  function computeRisk(raw) {
    const x = raw.map(function (v, i) {
      v = parseFloat(v);
      if (isNaN(v)) return MODEL.median[i];
      return Math.max(RANGES[i][0], Math.min(RANGES[i][1], v));
    });
    const contrib = x.map(function (v, i) { return MODEL.coef[i] * (v - MODEL.mean[i]) / MODEL.scale[i]; });
    const z = MODEL.intercept + contrib.reduce(function (a, b) { return a + b; }, 0);
    return { p: 1 / (1 + Math.exp(-z)), contrib: contrib, x: x };
  }
  /*PREDICT_END*/

  function readInputs() {
    const g = function (id) { return document.getElementById(id).value; };
    return [g('pregInput'), g('glucoseSlider'), g('bpInput'), g('skinInput'), g('insulinSlider'), g('bmiInput'), g('dpfSlider'), g('ageInput')];
  }

  function switchSection(section) {
    const demTab = document.getElementById('tabDemographics');
    const vitTab = document.getElementById('tabVitals');
    const genTab = document.getElementById('tabGenetics');
    const demSec = document.getElementById('sec-demographics');
    const vitSec = document.getElementById('sec-vitals');
    const genSec = document.getElementById('sec-genetics');

    const inactiveClass = 'flex-1 py-2 px-2.5 rounded-lg font-label-md text-label-md text-center transition-all text-on-surface-variant hover:text-on-surface flex items-center justify-center gap-1';
    const activeClass = 'flex-1 py-2 px-2.5 rounded-lg font-label-md text-label-md text-center transition-all bg-surface-container-lowest text-primary shadow-sm font-semibold flex items-center justify-center gap-1';

    demTab.className = inactiveClass;
    vitTab.className = inactiveClass;
    genTab.className = inactiveClass;

    if (section === 'demographics') {
      demTab.className = activeClass;
      demSec.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } else if (section === 'vitals') {
      vitTab.className = activeClass;
      vitSec.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } else if (section === 'genetics') {
      genTab.className = activeClass;
      genSec.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }

  function adjustValue(elementId, delta) {
    const el = document.getElementById(elementId);
    let val = parseFloat(el.value) || 0;
    const min = parseFloat(el.min);
    const max = parseFloat(el.max);
    val = Math.max(min, Math.min(max, val + delta));
    el.value = val;
    if (elementId === 'ageInput') {
      document.getElementById('ageSlider').value = val;
      updateAgeCohort(val);
    }
  }

  function syncAgeFromSlider(val) {
    document.getElementById('ageInput').value = val;
    updateAgeCohort(val);
  }

  function syncAgeFromNum(val) {
    const clamped = Math.max(21, Math.min(81, val));
    document.getElementById('ageSlider').value = clamped;
    updateAgeCohort(clamped);
  }

  function updateAgeCohort(val) {
    const label = document.getElementById('ageCohortLabel');
    if (val < 30) label.textContent = 'Young Adult (' + val + ' yr)';
    else if (val < 50) label.textContent = 'Adult (' + val + ' yr)';
    else if (val < 65) label.textContent = 'Mature Adult (' + val + ' yr)';
    else label.textContent = 'Senior (' + val + ' yr)';
  }

  function updateBmiState(val) {
    const num = parseFloat(val);
    const badge = document.getElementById('bmiClassBadge');
    const meter = document.getElementById('bmiMeterBar');
    const pct = Math.min(100, Math.max(0, ((num - 18) / (55 - 18)) * 100));
    meter.style.width = pct + '%';

    if (num < 18.5) {
      badge.textContent = 'Underweight';
      badge.className = 'font-label-sm text-label-sm px-2.5 py-0.5 rounded-full bg-surface-container-high text-on-surface-variant font-semibold';
      meter.className = 'h-full bg-outline-variant transition-all';
    } else if (num < 25) {
      badge.textContent = 'Normative';
      badge.className = 'font-label-sm text-label-sm px-2.5 py-0.5 rounded-full bg-secondary-container text-on-secondary-container font-semibold';
      meter.className = 'h-full bg-secondary transition-all';
    } else if (num < 30) {
      badge.textContent = 'Overweight';
      badge.className = 'font-label-sm text-label-sm px-2.5 py-0.5 rounded-full bg-tertiary-fixed text-on-tertiary-fixed-variant font-semibold';
      meter.className = 'h-full bg-primary transition-all';
    } else {
      badge.textContent = 'Obese Range';
      badge.className = 'font-label-sm text-label-sm px-2.5 py-0.5 rounded-full bg-tertiary text-on-tertiary font-semibold';
      meter.className = 'h-full bg-tertiary transition-all';
    }
  }

  function setBmiPreset(val) {
    const input = document.getElementById('bmiInput');
    input.value = val;
    updateBmiState(val);
  }

  function toggleBmiInfo() {
    document.getElementById('bmiInfoCard').classList.toggle('hidden');
  }

  function togglePedigreeInfo() {
    document.getElementById('dpfInfoCard').classList.toggle('hidden');
  }

  function updateGlucoseDisplay(val) {
    document.getElementById('glucoseValueDisplay').textContent = val;
    const tag = document.getElementById('glucoseStatusTag');
    const num = parseInt(val, 10);
    if (num < 140) {
      tag.textContent = 'Normal (' + num + ')';
      tag.className = 'text-secondary font-semibold';
    } else if (num < 200) {
      tag.textContent = 'Impaired Tolerance (' + num + ')';
      tag.className = 'text-tertiary font-semibold';
    } else {
      tag.textContent = 'Diabetic Range (200+)';
      tag.className = 'text-tertiary font-bold';
    }
  }

  function updateInsulinDisplay(val) {
    document.getElementById('insulinValText').textContent = val;
    document.getElementById('insulinNum').value = val;
  }

  function syncInsulinSlider(val) {
    const clamped = Math.max(15, Math.min(450, val));
    document.getElementById('insulinSlider').value = clamped;
    document.getElementById('insulinValText').textContent = clamped;
  }

  function updateDpfDisplay(val) {
    document.getElementById('dpfScore').textContent = parseFloat(val).toFixed(2);
  }

  function setDpf(val) {
    document.getElementById('dpfSlider').value = val;
    updateDpfDisplay(val);
  }

  function resetAssessmentForm() {
    document.getElementById('ageInput').value = 33;
    document.getElementById('ageSlider').value = 33;
    updateAgeCohort(33);

    document.getElementById('bmiInput').value = 26.4;
    updateBmiState(26.4);

    document.getElementById('pregInput').value = 1;
    document.getElementById('glucoseSlider').value = 115;
    updateGlucoseDisplay(115);

    document.getElementById('bpInput').value = 76;
    document.getElementById('skinInput').value = 25;

    document.getElementById('insulinSlider').value = 85;
    document.getElementById('insulinNum').value = 85;
    document.getElementById('insulinValText').textContent = 85;

    document.getElementById('dpfSlider').value = 0.47;
    updateDpfDisplay(0.47);

    closeResultModal();
  }

  function showResult(r) {
    const pct = Math.round(r.p * 100);
    let label, color, wrap, icon;
    if (r.p < 0.25) { label = 'Lower Metabolic Risk'; color = '#006c4a'; wrap = '#82f5c1'; icon = 'check_circle'; }
    else if (r.p < 0.5) { label = 'Moderate Metabolic Risk'; color = '#8a5a00'; wrap = '#ffe2a8'; icon = 'info'; }
    else if (r.p < 0.75) { label = 'Higher Metabolic Risk'; color = '#ba0035'; wrap = '#ffdada'; icon = 'warning'; }
    else { label = 'High Metabolic Risk'; color = '#ba0035'; wrap = '#ffdada'; icon = 'warning'; }

    const pctEl = document.getElementById('resPct');
    pctEl.textContent = pct + '%';
    pctEl.style.color = color;
    const lab = document.getElementById('resLabel');
    lab.textContent = label;
    lab.style.color = color;
    document.getElementById('resIconWrap').style.background = wrap;
    const ic = document.getElementById('resIcon');
    ic.textContent = icon;
    ic.style.color = color;

    // Largest contributor vs. the average patient in the training data
    const order = r.contrib.map(function (c, i) { return [c, i]; }).sort(function (a, b) { return b[0] - a[0]; });
    const top = order[0];
    let txt = 'Estimated chance of diabetes from a logistic regression model trained on 768 patient records (about 77% cross-validated accuracy). ';
    if (top[0] > 0.05) txt += FEATURES[top[1]] + ' is the biggest factor raising this estimate compared with the average patient. ';
    else txt += 'No single input is pushing this estimate above the average patient. ';
    txt += 'This is a screening aid, not a diagnosis.';
    document.getElementById('resText').textContent = txt;

    const mx = Math.max(0.5, Math.abs(order[0][0]), Math.abs(order[order.length - 1][0]));
    document.getElementById('resBars').innerHTML = '<span class="font-label-sm text-label-sm text-on-surface-variant">What is pushing your result (right raises risk, left lowers it)</span>' +
      order.map(function (o) {
        const w = Math.min(50, Math.abs(o[0]) / mx * 50);
        const bar = o[0] >= 0 ? 'left:50%;width:' + w + '%;background:#ba0035' : 'left:' + (50 - w) + '%;width:' + w + '%;background:#006c4a';
        return '<div class="flex items-center gap-2"><span class="font-body-sm text-body-sm text-on-surface-variant" style="width:96px">' + FEATURES[o[1]] + '</span>' +
          '<div class="relative flex-1 h-2.5 rounded-full bg-surface-container"><div class="absolute top-0 bottom-0 rounded-full" style="' + bar + '"></div><div class="absolute top-[-2px] bottom-[-2px] w-px bg-outline" style="left:50%"></div></div></div>';
      }).join('');
  }

  function triggerAssessment() {
    const btn = document.getElementById('runAssessmentBtn');
    const originalText = btn.innerHTML;
    btn.innerHTML = '<span class="material-symbols-outlined text-[20px] animate-spin">refresh</span><span>Computing risk...</span>';
    btn.disabled = true;

    setTimeout(function () {
      btn.innerHTML = originalText;
      btn.disabled = false;
      showResult(computeRisk(readInputs()));
      const modal = document.getElementById('resultModal');
      modal.classList.remove('hidden');
      modal.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 600);
  }

  function closeResultModal() {
    document.getElementById('resultModal').classList.add('hidden');
  }
</script><aside class="px-margin py-space-sm mx-margin my-space-md rounded-lg bg-surface-container-low"><p class="font-body-sm text-body-sm text-on-surface-variant"><strong class="text-primary font-semibold">CLINICAL DECISION SUPPORT:</strong> For investigational use alongside standard biometric verification. Not intended as sole primary diagnosis.</p></aside></main><nav class="fixed bottom-0 w-full z-50 pb-safe bg-surface/85 backdrop-blur-xl shadow-[0_-2px_12px_rgba(0,0,0,0.04)]" data-active-classes="text-primary font-semibold bg-surface-container-low"><div class="flex justify-around items-center h-16 px-space-xs"><a class="flex flex-col items-center justify-center gap-0.5 min-w-[64px] h-12 rounded-xl text-on-surface-variant hover:text-on-surface transition-colors" data-path="clinical-dashboard" href="#"><span class="material-symbols-outlined text-[22px]">space_dashboard</span><span class="font-label-sm text-label-sm">Dashboard</span></a><a aria-current="page" class="flex flex-col items-center justify-center gap-0.5 min-w-[64px] h-12 rounded-xl transition-colors text-primary font-semibold bg-surface-container-low" data-path="patient-assessment" href="#"><span class="material-symbols-outlined text-[22px]">analytics</span><span class="font-label-sm text-label-sm">Predict</span></a><a class="flex flex-col items-center justify-center gap-0.5 min-w-[64px] h-12 rounded-xl text-on-surface-variant hover:text-on-surface transition-colors" data-path="metabolic-insights" href="#"><span class="material-symbols-outlined text-[22px]">insights</span><span class="font-label-sm text-label-sm">Insights</span></a><a class="flex flex-col items-center justify-center gap-0.5 min-w-[64px] h-12 rounded-xl text-on-surface-variant hover:text-on-surface transition-colors" data-path="model-telemetry" href="#"><span class="material-symbols-outlined text-[22px]">neurology</span><span class="font-label-sm text-label-sm">Model Info</span></a></div></nav></body></html>"""

html_doc = "<!DOCTYPE html>" + HTML.replace("__MODEL__", MODEL_JS)

FRAME_HEIGHT = 900
if hasattr(st, "iframe"):            # newer Streamlit
    st.iframe(html_doc, height=FRAME_HEIGHT)
else:                                # older Streamlit
    import streamlit.components.v1 as components
    components.html(html_doc, height=FRAME_HEIGHT, scrolling=False)