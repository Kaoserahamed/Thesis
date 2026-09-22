# Repository Engineering Quality Note for AI Coding Agents

## Purpose

When building, refactoring, or documenting a project, follow this note to produce a professional, maintainable, reproducible repository. The guidance is based on recurring patterns observed in public DataFactor-style repository assessment pull requests and related example repositories.

> **Important:** This is an inferred checklist, not an official published DataFactor scoring rubric. Do not optimize by fabricating history, adding meaningless tests, or hiding weaknesses.

---

## Core Instructions for the AI Agent

Before writing code, understand the project requirements, architecture, technology stack, target users, and deployment environment. Build the project as a real maintainable software product—not as a demo or a collection of disconnected files.

Prioritize:

1. Correctness and meaningful functionality
2. Automated testing
3. Reproducible installation and execution
4. CI/CD validation
5. Security and clean configuration
6. Clear documentation
7. Maintainable architecture
8. Authentic, incremental development practices

---

## 1. Testing

- Use a real testing framework appropriate to the language and stack.
- Write meaningful behavioral tests with real assertions.
- Cover core business logic, API endpoints, validation, error cases, and important edge cases.
- Ensure tests can run from a fresh clone.
- Avoid tests that depend on live hardware, unavailable external services, or unstable network calls.
- Mock external services when appropriate.
- Do not create meaningless tests only to increase coverage.
- Add tests alongside new features and bug fixes.
- Track coverage where practical and set a reasonable project-specific coverage target.

---

## 2. CI/CD

Create a working GitHub Actions or equivalent CI pipeline.

At minimum, CI should:

1. Install dependencies
2. Validate dependency consistency
3. Run formatting checks
4. Run linting/static checks
5. Build the project
6. Run automated tests
7. Optionally run security/dependency audits

CI should run on pull requests and pushes to the main development branch. The pipeline must actually execute the checks and fail when a required check fails; do not create placeholder workflows.

---

## 3. Reproducible Builds

- Include the correct dependency manifest.
- Commit the appropriate lockfile whenever the ecosystem supports it.
- Pin or constrain dependencies responsibly.
- Ensure a fresh clone can be installed without undocumented manual steps.
- Use reproducible commands such as `npm ci` or an equivalent package-manager command when applicable.
- Document supported runtime versions, such as Node.js, Python, Java, or Docker versions.
- Verify installation and execution in a clean environment.

---

## 4. Documentation

Create a useful and accurate `README.md` containing:

- Project title and concise purpose
- Main features
- Technology stack
- System requirements
- Installation instructions
- Environment-variable setup
- Local development commands
- Test commands
- Build commands
- Deployment instructions or deployment overview
- Architecture overview
- API or route documentation when relevant
- Screenshots or demo links when useful
- Known limitations and future improvements

Also add, when appropriate:

- `.env.example`
- `CONTRIBUTING.md`
- `SECURITY.md`
- `CHANGELOG.md`
- `docs/` for architecture, API, deployment, database, and operational documentation
- Issue templates and pull-request templates

Never claim that a feature, test, deployment, or integration exists unless it has actually been implemented and verified.

---

## 5. Security and Configuration

- Never hardcode API keys, passwords, tokens, private keys, or cloud credentials.
- Read secrets from environment variables or a secure secret manager.
- Add `.env.example` with variable names, safe placeholder values, and short descriptions.
- Ensure `.env`, credential files, build outputs, logs, and other sensitive/generated files are excluded through `.gitignore`.
- Validate and sanitize external input.
- Use structured error responses without exposing secrets, stack traces, or internal implementation details.
- Apply authentication and authorization consistently when required.
- Add dependency/security scanning where practical.

---

## 6. Code Quality and Maintainability

- Follow the conventions of the chosen language and framework.
- Use clear naming and small, focused modules.
- Avoid unnecessary complexity and premature abstraction.
- Remove dead code, unused imports, commented-out implementation blocks, and debug statements.
- Replace raw `print`, `console.log`, or stack-trace output with appropriate structured logging in production code.
- Handle errors explicitly and consistently.
- Avoid duplicated classes, components, routes, and business logic.
- Extract genuinely shared functionality into reusable modules.
- Keep application code separate from generated files, vendored libraries, temporary files, and demo material.
- Keep the codebase understandable for a new contributor.

---

## 7. Project Structure

Use a structure appropriate to the stack, but keep responsibilities clearly separated. A typical structure may include:

```text
project-root/
├── src/ or app/
├── tests/
├── docs/
├── scripts/
├── config/
├── .github/
│   ├── workflows/
│   ├── ISSUE_TEMPLATE/
│   └── pull_request_template.md
├── .env.example
├── .gitignore
├── README.md
├── CONTRIBUTING.md
├── SECURITY.md
├── Dockerfile
├── docker-compose.yml
├── dependency-manifest
└── lockfile
```

Do not force every directory into a project if it is unnecessary. Prefer a small, coherent structure over empty or artificial folders.

---

## 8. Containerization

When relevant, provide:

- A production-appropriate `Dockerfile`
- A `.dockerignore`
- A `docker-compose.yml` or Compose configuration for local development
- Health checks where appropriate
- Separate development and production concerns when needed
- Non-root execution where practical
- Clear instructions for starting the full system

The application should be runnable with a documented command and should fail clearly when required configuration is missing.

---

## 9. Git History and Development Practice

- Make small, focused commits.
- Pair features with their tests where possible.
- Use meaningful commit messages.
- Do not fabricate, backdate, or artificially inflate commit history.
- Do not create hundreds of low-value tests or meaningless files to manipulate an assessment.
- Maintain a genuine development history with understandable milestones.
- Use branches and pull requests for significant changes.
- Keep the main branch protected when the project and team workflow support it.

---

## 10. Repository Governance

When applicable, configure:

- Protected `main` branch
- Required pull-request review
- Required CI checks
- Security checks such as CodeQL or dependency scanning
- Stale-review dismissal after new commits
- Disabled force-push and branch deletion on `main`

These settings may need to be configured in the Git hosting platform rather than inside the repository.

---

## 11. AI/ML and Data Science Project Requirements

For machine-learning or research repositories, additionally include:

- Clear problem statement and research objective
- Dataset description and data-source information
- Data-processing and preprocessing pipeline
- Reproducible experiment configuration
- Training, validation, and test separation
- Fixed random seeds where appropriate
- Model architecture explanation
- Evaluation metrics and baseline comparisons
- Training and inference instructions
- Model checkpoints or clear instructions for obtaining them
- Experiment results with limitations
- Data and model licensing considerations
- Notebook demonstrations separated from reusable source code
- Tests for preprocessing, data validation, model interfaces, and key utility functions
- Avoid committing large datasets or generated artifacts unless necessary and permitted

A notebook should demonstrate the work, while the main repository should contain clean, reusable, maintainable implementation code.

---

## 12. Final Verification Checklist

Before considering the project complete, the AI agent must:

- [ ] Verify the project structure
- [ ] Run formatting checks
- [ ] Run linting/static analysis
- [ ] Run the build
- [ ] Run automated tests
- [ ] Check test results and fix failures
- [ ] Verify dependency installation from a clean environment
- [ ] Check that no secrets are committed
- [ ] Validate `.env.example`
- [ ] Review error handling and logging
- [ ] Review duplicated or dead code
- [ ] Confirm README setup instructions work
- [ ] Confirm Docker setup works, if provided
- [ ] Confirm CI configuration is valid
- [ ] Review security and dependency risks
- [ ] Document known limitations honestly
- [ ] Report what was verified and what could not be verified

---

## Recommended Agent Behavior

When working on a project:

1. Inspect the existing repository before changing anything.
2. Create a short implementation plan.
3. Identify missing quality gates early.
4. Implement in small, logically separated steps.
5. Add tests with each meaningful feature.
6. Run validation commands after changes.
7. Fix errors instead of ignoring or bypassing them.
8. Keep documentation synchronized with the implementation.
9. Avoid unnecessary rewrites and preserve working functionality.
10. Clearly report assumptions, limitations, and unverified items.

The goal is a repository that is functional, testable, secure, reproducible, documented, and genuinely maintainable—not merely optimized for an automated score.
