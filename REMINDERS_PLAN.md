# Voice reminders for Nova

## What the first version should do

- Create one-time reminders by voice. “Remind me tomorrow at six to start the Minecraft server” should prompt for morning or evening, then save the reminder and read back the task and the full local date and time.
- List upcoming and snoozed reminders. Use the task name or a numbered choice to resolve ambiguous matches when canceling or snoozing.
- Deliver a Plasma desktop notification with a sound while the user is logged in, even when Nova is closed. Keep reminders due during shutdown or logout and show a catch-up summary at the next login. Deliver reminders due during suspend after resume.

## Implementation order

1. **Store reminders locally.** Add a small SQLite store under the same XDG state directory used by learned commands, in a separate database shared by Nova and the reminder service. Store an ID, task, due time in UTC, and delivery status. Use atomic updates to protect against concurrent cancellation, snoozing, or delivery. Keep a delivered reminder available for snoozing and set a new due time when snoozed.
2. **Add a local voice handler.** In `src/voice_assistant/main.py`, handle explicit reminder phrases after configured and learned command lookup and before the general agent fallback. Start with “in ten minutes,” “tomorrow at six PM,” and explicit dates and times. Interpret calendar times in the computer’s local time zone. Ask for clarification about missing times, ambiguous dates, or clock changes. Keep follow-up state during the voice session so answers like “PM” or “the second one” apply to the pending request. Read back successful changes only after the database update succeeds.
3. **Run delivery in a separate systemd user service.** Follow the graphical-session setup in `systemd/voice-assistant.service` and give the reminder service its own lifecycle. Check for due reminders on startup and at a short interval using the current clock. Use the desktop notification service for the visible alert and verify sound on Plasma. Leave reminders pending after notification failures and retry with a delay. Record delivery after the desktop accepts the notification. A crash between notification and recording delivery can cause a repeat, so test that case and favor a repeat over losing the reminder.
4. **Verify the feature on this computer.** Test parsing, local time and daylight-saving changes, persistence, and conflicting updates. Then create a short reminder by voice, hear the confirmation, stop Nova, and verify that the notification and sound still arrive. Restart Nova and verify voice listing, snoozing, and cancellation. Test suspend and logout catch-up, plus delivery after restarting the reminder service. Document how to enable and inspect the service.

Start with one-time reminders and voice-based snoozing and cancellation.
