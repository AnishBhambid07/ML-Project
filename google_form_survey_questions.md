# Employee Attrition & Workforce Sentiment Survey
## Google Form Manual Data Collection Guide (ML CA1 & CA2 Project)

This document contains the exact questions, response options, validation rules, and feature mapping used for collecting empirical employee data via Google Forms.

---

### Survey Setup Instructions for Google Forms
1. Open [Google Forms](https://forms.google.com/) and create a new blank form.
2. Form Title: `Workplace Sentiment, Compensation & Employee Retention Study`
3. Description:
   > *"This anonymous academic survey is conducted for an Machine Learning Research Project on predictive workplace retention dynamics. All responses are strictly confidential and used solely for algorithmic feature modeling and statistical evaluation."*
4. Enable *"Collect email addresses: OFF"* to ensure candid responses.

---

### Section 1: Demographics & Career Tenure

#### 1. What is your current age?
- **Type:** Short Answer (Number validation: Integer between 18 and 65)
- **Feature Mapped:** `Age`
- **Required:** Yes

#### 2. What is your gender identity?
- **Type:** Multiple Choice
  - `Female`
  - `Male`
  - `Prefer not to say`
- **Feature Mapped:** `Gender`
- **Required:** Yes

#### 3. What is your current marital status?
- **Type:** Multiple Choice
  - `Single`
  - `Married`
  - `Divorced`
- **Feature Mapped:** `MaritalStatus`
- **Required:** Yes

#### 4. How many total years of professional working experience do you have?
- **Type:** Short Answer (Number validation: Integer 0 to 45)
- **Feature Mapped:** `TotalWorkingYears`
- **Required:** Yes

#### 5. How many years have you been working at your current organization?
- **Type:** Short Answer (Number validation: Integer 0 to 40)
- **Feature Mapped:** `YearsAtCompany`
- **Required:** Yes

---

### Section 2: Organizational Unit & Compensation

#### 6. Which department best describes your business function?
- **Type:** Dropdown / Multiple Choice
  - `Research & Development / Engineering`
  - `Sales & Business Development`
  - `Human Resources`
- **Feature Mapped:** `Department`
- **Required:** Yes

#### 7. What is your specific job title / role category?
- **Type:** Dropdown
  - `Software Engineer / Developer`
  - `Research Scientist`
  - `Sales Executive`
  - `Laboratory Technician`
  - `Manufacturing Director / Lead`
  - `Healthcare Representative`
  - `Manager / Department Head`
  - `Sales Representative`
  - `HR Specialist`
- **Feature Mapped:** `JobRole`
- **Required:** Yes

#### 8. What is your approximate gross monthly compensation (in USD or INR)?
- **Type:** Multiple Choice (Binned) or Short Answer
  - `Under ₹2,00,000 / $2,500 per month` (Entry Level)
  - `₹2,00,000 - ₹4,00,000 / $2,500 - $5,000 per month` (Mid-Level)
  - `₹4,00,000 - ₹7,00,000 / $5,000 - $8,500 per month` (Senior)
  - `Above ₹7,00,000 / $8,500+ per month` (Leadership / Executive)
- **Feature Mapped:** `MonthlyIncome`
- **Required:** Yes

#### 9. What is your stock options / equity participation tier?
- **Type:** Multiple Choice
  - `Level 0 - No Stock Options`
  - `Level 1 - Modest Annual Grants`
  - `Level 2 - Substantial Equity Plan`
  - `Level 3 - Executive Keyholder Vesting`
- **Feature Mapped:** `StockOptionLevel`
- **Required:** Yes

---

### Section 3: Work Environment, Strain & Attrition Propensity

#### 10. Does your role frequently require overtime or weekend work?
- **Type:** Multiple Choice
  - `No - Standard 40 Hours Schedule`
  - `Yes - Regular Overtime / Extended Hours`
- **Feature Mapped:** `OverTime`
- **Required:** Yes

#### 11. Rate your current Job Satisfaction:
- **Type:** Linear Scale (1 to 4)
  - `1 = Very Low / Dissatisfied`
  - `2 = Low / Neutral`
  - `3 = High / Satisfied`
  - `4 = Very High / Fulfilled`
- **Feature Mapped:** `JobSatisfaction`
- **Required:** Yes

#### 12. Rate your current Work-Life Balance:
- **Type:** Linear Scale (1 to 4)
  - `1 = Poor / Significant Burnout`
  - `2 = Fair / Frequent Strain`
  - `3 = Good / Manageable`
  - `4 = Excellent / Well Balanced`
- **Feature Mapped:** `WorkLifeBalance`
- **Required:** Yes

#### 13. Approximately how many kilometers is your one-way daily commute?
- **Type:** Short Answer (Number 1 to 50 km)
- **Feature Mapped:** `DistanceFromHome`
- **Required:** Yes

#### 14. Target Variable: Are you actively planning or have you left an organization in the past 12 months?
- **Type:** Multiple Choice
  - `Yes (Exited / Actively Planning to Exit)`
  - `No (Retained / Intend to Stay Long-Term)`
- **Feature Mapped:** `Attrition` (Target Variable: `1` vs `0`)
- **Required:** Yes

---

### How to Import Google Form Responses into the Project
1. In Google Forms, click on the **Responses** tab & click **"Link to Sheets"**.
2. Download the Google Sheet as **CSV** (`File > Download > Comma Separated Values (.csv)`).
3. Open our website interface:
   - Navigate to the **Data Explorer** tab.
   - Click **"📋 Open Employee Survey / Google Form Data Collector"**.
   - Input the student/peer responses into the form modal to instantly view live multi-model predictions and test the newly collected records!
