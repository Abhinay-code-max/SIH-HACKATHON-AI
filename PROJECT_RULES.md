# CRITICAL PROJECT RULES

## RULE 1 — PROMPT/COMMAND-ONLY WORKFLOW

You are an AI development assistant for this project.

You MUST NOT assume that you can directly execute terminal commands, modify the user's environment, install packages, start servers, download files, or verify runtime behavior.

Your responsibility is to **generate prompts, terminal commands, code instructions, and verification steps only**.

The developer will:

1. Receive your instructions.
2. Copy/run the commands manually in their terminal.
3. Inspect the output.
4. Test the application.
5. Return the results/errors to you.
6. You then generate the next set of commands/instructions.

### Therefore:

DO NOT say:
* "I installed..."
* "I ran..."
* "The server is now running..."
* "I verified..."
* "The model works..."
* "I fixed the issue..."

unless the developer has explicitly provided terminal output proving it.

Instead say:
> "Run the following command and send me the output."
or:
> "Apply the following change, then run these verification commands."

Every implementation step should contain:
### 1. ACTION
What needs to be changed.
### 2. TERMINAL COMMAND
The exact command(s) the developer should run.
### 3. EXPECTED RESULT
What should appear if successful.
### 4. VERIFICATION
How the developer should verify it manually.
### 5. NEXT STEP
What should be done after successful verification.

Do not combine dozens of unrelated changes into one enormous command.
Prefer small, independently verifiable milestones.

---

# RULE 2 — ABSOLUTE OFFLINE OPERATION

The ENTIRE PROJECT must be capable of running with **ZERO INTERNET CONNECTION**.

This is a hard architectural requirement, not an optional optimization.

The final prototype must work when:
* Wi-Fi is disabled.
* Ethernet is disconnected.
* Internet access is blocked.
* DNS is unavailable.

The application must still launch and operate.

This applies to:
* Frontend
* Backend
* AI inference
* Computer vision
* Mapping
* Local data
* Model loading
* Assets
* Fonts
* Icons
* Video processing
* Event generation
* Storage
* API communication

---

# OFFLINE DEVELOPMENT VS OFFLINE RUNTIME

Internet may be used during DEVELOPMENT to obtain:
* Python packages
* Node packages
* AI model weights
* Map datasets
* Documentation
* Development dependencies

However, once these resources are installed/prepared, the actual application MUST NOT require the internet.

Clearly separate:
### Development-time dependencies
from
### Runtime dependencies

The runtime dependency list must contain only resources available locally.

---

# NO CLOUD FALLBACKS

Do NOT silently add cloud fallbacks.

For example, NEVER implement:
```text
Local AI
   ↓
If unavailable
   ↓
Cloud AI
```

Instead:
```text
Local AI
   ↓
If unavailable
   ↓
Clear local error/status
```

The same applies to:
* Maps
* Geocoding
* Routing
* Databases
* AI APIs
* Image processing
* Speech processing
* Analytics

---

# NO ONLINE FRONTEND RESOURCES

The frontend must not load:
* Google Fonts
* CDN JavaScript
* CDN CSS
* Remote icons
* Remote images
* Remote map tiles
* Analytics scripts
* External APIs
during runtime.

Everything required by the frontend must be bundled locally.

---

# OFFLINE AI

All AI inference must occur locally.

The AI pipeline should be:
```text
Local Camera / Local Video
          ↓
Local Preprocessing
          ↓
Local AI Model
          ↓
Local Detection
          ↓
Local Event Engine
          ↓
Local Backend
          ↓
Local Frontend
```

No external AI API should be required.
Model weights must be available locally.
The application must explicitly report when a model is missing rather than attempting to download it automatically.

---

# OFFLINE MAPPING

Maps must also work without internet.
Do not use an online-only mapping architecture as the foundation.
Use locally stored map data.
The prototype should package only the geographic region required for the demonstration.

No runtime requests for:
* Map tiles
* Geocoding
* Directions
* Places
* Routing
* Satellite imagery
unless the required data is already stored locally.

---

# OFFLINE VERIFICATION IS MANDATORY

Before declaring Phase 1 complete, perform an actual offline test.

The developer should:
1. Install all required dependencies.
2. Prepare all models/assets/map data.
3. Start the application normally.
4. Verify the complete pipeline.
5. Disconnect the machine from the internet.
6. Restart the entire project.
7. Run the same workflow again.

The expected result is:
```text
Internet: OFF
       ↓
Frontend starts
       ↓
Backend starts
       ↓
AI model loads
       ↓
Local video/camera works
       ↓
AI inference works
       ↓
Events generated
       ↓
Offline map loads
       ↓
Frontend receives local data
       ↓
Complete prototype works
```

If anything attempts to access the internet, that is considered a **Phase 1 failure**.

---

# OFFLINE DEPENDENCY AUDIT

Before Prototype v0.1 is frozen, generate a dependency audit containing:

| Component | Dependency | Local? | Internet Required at Runtime? |
| --------- | ---------- | ------ | ----------------------------- |
| Frontend  | ...        | YES/NO | YES/NO                        |
| Backend   | ...        | YES/NO | YES/NO                        |
| AI        | ...        | YES/NO | YES/NO                        |
| Model     | ...        | YES/NO | YES/NO                        |
| Mapping   | ...        | YES/NO | YES/NO                        |
| Assets    | ...        | YES/NO | YES/NO                        |

There must be **zero runtime internet dependencies**.

---

# DEVELOPMENT STYLE

Work incrementally.
Do NOT generate the entire project at once.

Use this progression:
```text
STEP 1: Environment verification
        ↓
STEP 2: Local AI model loading
        ↓
STEP 3: Image inference
        ↓
STEP 4: Video inference
        ↓
STEP 5: Detection schema
        ↓
STEP 6: Event engine
        ↓
STEP 7: Offline mapping
        ↓
STEP 8: Local backend
        ↓
STEP 9: Frontend integration
        ↓
STEP 10: End-to-end testing
        ↓
STEP 11: Internet disabled test
        ↓
STEP 12: Prototype v0.1
```

After every major step:
**STOP → provide verification commands → wait for the developer's result.**
Do not continue assuming that the previous step succeeded.

---

# WHEN ERRORS OCCUR

When the developer provides an error:
1. Analyze the exact error.
2. Explain the likely cause briefly.
3. Provide the smallest fix.
4. Provide exact terminal commands.
5. Provide verification commands.
6. Wait for the result.

Do not rewrite unrelated parts of the project to fix a localized problem.

---

# FINAL PROJECT PRINCIPLE

The prototype should be treated as a **self-contained local system**.

The ultimate test is:
> "Take the completed project to a machine with no internet connection and run the prototype."
If it cannot do that, the architecture is not finished.
