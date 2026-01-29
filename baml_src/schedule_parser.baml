// baml_src/schedule_parser.baml

class HSDSSchedule {
  // Core Fields
  opens_at string? @description("HH:MM:SS format. The time when a service or location opens.")
  closes_at string? @description("HH:MM:SS format. The time when a service or location closes.")
  freq string? @description("DAILY, WEEKLY, MONTHLY, YEARLY. How often the frequency repeats.")
  interval int? @description("Default 1. How often the frequency repeats (e.g. 2 for every other week).")
  byday string? @description("Comma separated days: MO,TU,WE,TH,FR,SA,SU. Can have position for MONTHLY (e.g. 1MO, -1FR).")
  bymonthday string? @description("Comma separated days of month: 1, 2, ... 31. Can be negative.")
  dtstart string? @description("YYYY-MM-DD. Start date of the schedule.")
  until string? @description("YYYY-MM-DD. End date of the recurrence.")
  
  // New/Extended Fields
  valid_from string? @description("YYYY-MM-DD. Date from which the schedule information is valid.")
  valid_to string? @description("YYYY-MM-DD. Last date on which the schedule information is valid.")
  timezone int? @description("UTC offset (e.g. -5, 3).")
  count int? @description("Number of times the event occurs.")
  wkst string? @description("MO, TU, etc. Day on which the week starts.")
  byweekno string? @description("Comma separated week numbers.")
  byyearday string? @description("Comma separated year days.")
  schedule_link string? @description("URL for the schedule.")
  attending_type string? @description("Free text description of how to attend.")
  
  // Descriptive Fields
  notes string? @description("Free text notes, e.g. 'Closed on holidays'.")
}

function ParseSchedule(schedule_text: string) -> HSDSSchedule[] {
  client CustomGPT5Mini
  prompt #"
    Parse the schedule text into a LIST of HSDS Schedule objects.
    Each object represents a distinct time rule (RFC 5545 RRULE concept).
    Write using simple language, no higher than a fifth grade reading level.
    
    CRITICAL: 
    1. If the text describes multiple distinct time slots (e.g. "Meals: M-F. Office: Sat"), SPLIT them into separate objects.
    2. If NO specific schedule (days, times, frequency) can be extracted (e.g. "Class times vary", "Call for appointment", "Closed temporarily"), return an EMPTY LIST [].

    Schema Mapping:
    - opens_at/closes_at: HH:MM:SS (24hr).
    - freq: DAILY, WEEKLY, MONTHLY, YEARLY.
    - byday: MO,TU,WE,TH,FR,SA,SU. For MONTHLY, can include position: 1MO (1st Mon), 3SA (3rd Sat), -1FR (last Fri).
    - bymonthday: 1 to 31.
    - interval: e.g. 1 (every), 2 (every other).
    - notes: Short advisories (e.g. "Closed on holidays"). NO reasoning.

    Examples:

    In: "Monday-Friday 9am-5pm"
    Out: [
      {
        "opens_at": "09:00:00", "closes_at": "17:00:00",
        "freq": "WEEKLY", "interval": 1,
        "byday": "MO,TU,WE,TH,FR"
      }
    ]

    In: "Class times vary. Call for details."
    Out: []

    In: "Meals: W Th. Office: M-Th, 8:30am-5pm."
    Out: [
      {
        "freq": "WEEKLY", "byday": "WE,TH",
        "notes": "Meals"
      },
      {
        "opens_at": "08:30:00", "closes_at": "17:00:00",
        "freq": "WEEKLY", "byday": "MO,TU,WE,TH"
      }
    ]

    In: "9:00am - 10:30am, 3rd Saturday of each month\nCall for emergency needs"
    Out: [
      {
        "opens_at": "09:00:00", "closes_at": "10:30:00",
        "freq": "MONTHLY", "interval": 1,
        "byday": "3SA",
        "notes": "Call for emergency needs"
      }
    ]

    In: "Mo-Th 8am - 5pm, Fr 8am - 3pm"
    Out: [
      {
        "opens_at": "08:00:00", "closes_at": "17:00:00",
        "freq": "WEEKLY", "byday": "MO,TU,WE,TH"
      },
      {
        "opens_at": "08:00:00", "closes_at": "15:00:00",
        "freq": "WEEKLY", "byday": "FR"
      }
    ]

    Parse:
    {{ schedule_text }}

    {{ ctx.output_format }}
  "#
}