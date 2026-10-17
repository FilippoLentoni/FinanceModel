# Beta EC2 operator access

`config/beta.json` configures `dev_desktop_access.role_names` with the exact IAM role name `finplan-devdesktop`. This admits the development desktop to the research and registry buckets, job-control table, and job API. API handler authorization also recognizes the role for cancellation, strategy selection, and approval of another principal's run.

Gamma and prod reject nonempty operator exceptions. Policy builders also ignore supplied development-role exceptions outside beta. Role names cannot contain wildcards or resource ARNs.

The exception changes access allowlists. Registry conditional-create/no-delete policies, TLS, other-environment denials, explicit user confirmation for strategy selection, submitter/approver separation, lease limits, and cost/budget checks remain effective. It does not auto-approve a run, launch compute, or select a strategy.
