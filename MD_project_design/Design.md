# Financial Crime Investigation AI --- Design System

## 1. Design Direction

The product should feel like an enterprise financial-risk investigation
console.

Style: - professional - analytical - calm - information-dense -
trustworthy - minimal decoration

Avoid: - flashy AI gradients - excessive animations - chatbot-first
layout - game-like risk indicators - red everywhere

## 2. Theme

Primary theme: - dark-first - optional light mode

Background: - deep neutral/navy

Surface: - slightly lighter neutral

Primary accent: - blue

Secondary accent: - teal

Warning: - amber

Critical: - restrained red

Success: - green

Do not use color as the only indicator of state.

Every state should have: - icon - text label - color

## 3. Typography

Use: - Inter for UI - JetBrains Mono for IDs, code, event IDs and
technical values

Hierarchy:

``` text
Page title: 28–32px
Section title: 18–22px
Body: 14–16px
Metadata: 12–13px
```

## 4. Main Navigation

``` text
Overview
Investigations
Alerts
Transactions
Evidence Graph
XAI
Agents
Benchmarks
System Health
Settings
```

## 5. Investigation Workspace

The main screen should prioritize evidence.

``` text
+------------------------------------------------------+
| Case #AML-10291                 HIGH RISK   REVIEW   |
+------------------------------------------------------+
| Risk Score | Model | Timestamp | Status             |
+------------------------------------------------------+
| SHAP Explanation          | Evidence Summary       |
|                          |                        |
| Feature  Contribution    | Observed facts        |
| Amount    +0.41          | Transaction history   |
| Benef.    +0.22          | Device history       |
| Device    +0.17          | Beneficiary history  |
+--------------------------+-------------------------+
| Evidence Graph                                      |
|                                                     |
| Account A ---- Device D ---- Account C              |
|     |                                               |
|     +---------- Beneficiary B                       |
+------------------------------------------------------+
| Agent Investigation Timeline                        |
| Tool call -> Evidence -> Hypothesis -> Verification|
+------------------------------------------------------+
| Counterfactual | Missing Evidence | Human Decision |
+------------------------------------------------------+
```

## 6. Risk Presentation

Never show only:

``` text
94% FRAUD
```

Show:

``` text
94% model risk score
Requires human investigation
```

Include: - model version - threshold version - timestamp - explanation
availability

## 7. SHAP Visualization

Use: - horizontal bar chart - positive/negative contribution - exact
numeric contribution - feature value beside feature name

Example:

``` text
transaction_amount  +0.41  value=850000
new_beneficiary     +0.22  value=true
device_anomaly      +0.17  value=high
account_age         -0.03  value=4 years
```

Do not make SHAP values look like probabilities.

## 8. Evidence Cards

Every evidence card should contain:

``` text
Evidence ID
Type
Timestamp
Source
Observed value
Relationship to case
```

Example:

``` text
EV-001
Device relationship

Device D91 was associated with
3 accounts during the previous
30 days.

Source: device_history
```

## 9. Agent Timeline

Show:

``` text
19:41:02 Investigator Agent
19:41:03 Transaction Agent -> get_transaction_history
19:41:03 Tool result -> 12 transactions
19:41:04 Network Agent -> find_related_accounts
19:41:04 Tool result -> 4 related accounts
19:41:05 XAI Agent -> get_shap_explanation
19:41:05 Verification Agent -> PASS
```

Do not expose hidden chain-of-thought.

Show: - action - tool - result summary - evidence IDs - verification
status

## 10. Graph Design

Nodes: - account - transaction - device - beneficiary - merchant

Edges: - owns - uses - sends_to - receives_from - transacts_with

Use restrained highlighting for suspicious relationships.

## 11. Dashboard

Top cards:

``` text
Open Investigations
High-Risk Alerts
Cases Awaiting Review
Model PR-AUC
Evidence Accuracy
Agent Task Completion
```

Second row: - alert trend - risk distribution - SHAP feature
importance - investigation latency

## 12. Accessibility

-   keyboard navigation
-   sufficient contrast
-   labels for charts
-   do not rely on color alone
-   responsive layout
-   readable at 100% browser zoom

## 13. Empty/Error States

Never show blank screens.

Examples:

``` text
SHAP unavailable
The model prediction is available, but an explanation
could not be generated. No explanation was fabricated.
```

``` text
LLM unavailable
The deterministic investigation completed successfully.
AI-generated narrative is currently unavailable.
```

## 14. Frontend Rules

Keep the frontend separate from business logic.

The frontend consumes typed API responses.

Never calculate: - fraud score - SHAP contribution - evidence
relationships

in the browser.
