# What we collect

Flo is an open-source assistant run by the Kubeflow community. To find and fix
wrong answers, it records conversations. The chat widget says so under the input
box:

> Conversations and feedback are recorded to improve answer quality, so please
> don't share personal information.

## What is recorded

| Data | Why |
|---|---|
| Your questions and Flo's answers, with the searches and sources behind each answer | To see where retrieval or the answer went wrong |
| 👍 / 👎 votes and the optional comment | To find bad answers and route them to the right maintainers |
| Timing and token counts | To keep the service fast and within its GPU budget |

## What is not recorded

- **No accounts.** The widget gets an anonymous session token that expires after
  30 minutes. It identifies a chat, not a person.
- **No names or emails.** Nothing ties a conversation to who asked it, unless
  you type it into the chat yourself.

## Who can see it and for how long

- Conversations and feedback are visible only to the project's maintainers and
  mentors, in a self-hosted [Langfuse](https://langfuse.com) instance in the
  project's own cluster.
- They are **deleted after 4 weeks** by a weekly job. Evaluation runs, which use
  the project's own test questions and no user data, are deleted after 2 weeks.

Details for operators: [Observability](operations/observability.md#data-retention).
