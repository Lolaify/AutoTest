import os
import random
import pandas as pd
from faker import Faker

fake = Faker()
random.seed(42)
Faker.seed(42)

BASE_DIR = "/raid0_ssd2/pavel/AutoMatelda/datasets/synthetic_tables"
os.makedirs(BASE_DIR, exist_ok=True)

# ============================================================
# Helper Functions
# ============================================================

def inject_errors(df, error_rules, max_error_fraction=0.05):
    """
    Inject errors into dataframe columns.
    Default error ratio is 5% per SDC-covered column.
    Keeps error ratio well under 20% so SDCs still trigger reliably.
    """
    dirty_df = df.copy()

    for col, generator in error_rules.items():
        n_rows = len(df)
        n_errors = max(1, int(n_rows * max_error_fraction))
        indices = random.sample(range(n_rows), n_errors)

        for idx in indices:
            dirty_df.loc[idx, col] = generator()

    return dirty_df


def save_dataset(name, clean_df, dirty_df):
    dataset_dir = os.path.join(BASE_DIR, name)
    os.makedirs(dataset_dir, exist_ok=True)

    clean_df.to_csv(os.path.join(dataset_dir, "clean.csv"), index=False)
    dirty_df.to_csv(os.path.join(dataset_dir, "dirty.csv"), index=False)

    print(f"Saved dataset: {name}")


# ============================================================
# Dataset 1: Movies
# SDC Type: CTA (movie-related semantic domains)
# ============================================================

movies_clean = pd.DataFrame({
    "movie_title": [fake.sentence(nb_words=3).replace('.', '') for _ in range(100)],
    "director": [fake.name() for _ in range(100)],
    "genre": [random.choice(["Action", "Drama", "Comedy", "Sci-Fi", "Thriller"]) for _ in range(100)],
    "release_year": [str(random.randint(1980, 2023)) for _ in range(100)],
    "rating": [str(round(random.uniform(1, 10), 1)) for _ in range(100)]
})

movies_dirty = inject_errors(
    movies_clean,
    {
        "movie_title": lambda: random.choice(["123###", "!!!", "Unknown_Object"]),
        "director": lambda: random.choice(["Director404", "@@@@", "12345"]),
        "genre": lambda: random.choice(["Banana", "Toaster", "PlanetX"]),
        "release_year": lambda: random.choice(["YearUnknown", "Ancient", "-500"]),
        "rating": lambda: random.choice(["15.7", "-3.0", "Excellent"])
    }
)

save_dataset("movies", movies_clean, movies_dirty)


# ============================================================
# Dataset 2: Soccer Clubs
# SDC Type: CTA (team classifier)
# ============================================================

clubs = [
    "FC Barcelona", "Real Madrid", "Manchester United", "Bayern Munich",
    "Juventus", "Liverpool", "Arsenal", "Chelsea"
]

soccer_clean = pd.DataFrame({
    "club_name": [random.choice(clubs) for _ in range(100)],
    "country": [random.choice(["Spain", "England", "Germany", "Italy"]) for _ in range(100)],
    "stadium_capacity": [str(random.randint(20000, 90000)) for _ in range(100)],
    "coach": [fake.name() for _ in range(100)],
    "league": [random.choice(["La Liga", "Premier League", "Serie A", "Bundesliga"]) for _ in range(100)]
})

soccer_dirty = inject_errors(
    soccer_clean,
    {
        "club_name": lambda: random.choice(["Potato Corp", "Blue Chair", "Galaxy Soup"]),
        "country": lambda: random.choice(["Mars", "UnknownLand", "Kitchen"]),
        "stadium_capacity": lambda: random.choice(["-1000", "Huge", "999999999"]),
        "coach": lambda: random.choice(["@@@", "CoachBot3000", "1234"]),
        "league": lambda: random.choice(["Banana League", "Space Division", ""])
    }
)

save_dataset("soccer_clubs", soccer_clean, soccer_dirty)


# ============================================================
# Dataset 3: Countries
# SDC Type: CTA (country classifier)
# ============================================================

countries = [
    "Germany", "France", "Brazil", "Canada", "Japan",
    "India", "Australia", "Norway"
]

countries_clean = pd.DataFrame({
    "country": [random.choice(countries) for _ in range(100)],
    "capital": [fake.city() for _ in range(100)],
    "population_millions": [str(round(random.uniform(1, 1400), 2)) for _ in range(100)],
    "continent": [random.choice(["Europe", "Asia", "South America", "North America", "Oceania"]) for _ in range(100)],
    "currency": [random.choice(["Euro", "Dollar", "Yen", "Rupee"]) for _ in range(100)]
})

countries_dirty = inject_errors(
    countries_clean,
    {
        "country": lambda: random.choice(["MoonBase", "Atlantis", "ChocolateLand"]),
        "capital": lambda: random.choice(["123City", "???", "UnknownCapital"]),
        "population_millions": lambda: random.choice(["-50", "Many", "99999"]),
        "continent": lambda: random.choice(["MiddleEarth", "Ocean", "Galaxy"]),
        "currency": lambda: random.choice(["Credits", "Bananas", "GoldPieces"])
    }
)

save_dataset("countries", countries_clean, countries_dirty)


# ============================================================
# Dataset 4: URLs
# SDC Type: Function (url validator)
# ============================================================
urls_clean = pd.DataFrame({
    "website": [fake.url() for _ in range(100)],
    "company": [fake.company() for _ in range(100)],
    "contact_email": [fake.email() for _ in range(100)],
    "industry": [random.choice(["Finance", "Retail", "Tech", "Health"]) for _ in range(100)],
    "employees": [str(random.randint(10, 5000)) for _ in range(100)]
})

urls_dirty = inject_errors(
    urls_clean,
    {
        "website": lambda: random.choice(["not_a_url", "htp:/broken", "www missing dot"]),
        "company": lambda: random.choice(["###", "123Company", ""]),
        "contact_email": lambda: random.choice(["broken-email", "@@@", "mail.com"]),
        "industry": lambda: random.choice(["AlienTech", "Magic", "UnknownIndustry"]),
        "employees": lambda: random.choice(["-5", "Thousands", "99999999"])
    }
)

save_dataset("company_websites", urls_clean, urls_dirty)


# ============================================================
# Dataset 5: Email Addresses
# SDC Type: Function (email validation)
# ============================================================

emails_clean = pd.DataFrame({
    "employee_name": [fake.name() for _ in range(100)],
    "email": [fake.email() for _ in range(100)],
    "department": [random.choice(["HR", "IT", "Finance", "Marketing"]) for _ in range(100)],
    "office_city": [fake.city() for _ in range(100)],
    "salary": [str(random.randint(30000, 120000)) for _ in range(100)]
})

emails_dirty = inject_errors(
    emails_clean,
    {
        "employee_name": lambda: random.choice(["12345", "@@@@", "No_Name"]),
        "email": lambda: random.choice(["missingatsign.com", "@@broken", "invalid-email"]),
        "department": lambda: random.choice(["UnknownDept", "???", "123"]),
        "office_city": lambda: random.choice(["LaptopCity", "Nowhere", "###"]),
        "salary": lambda: random.choice(["-10000", "High", "999999999"])
    }
)

save_dataset("employee_emails", emails_clean, emails_dirty)


# ============================================================
# Dataset 6: Time Values
# SDC Type: Pattern (time regex)
# ============================================================

schedule_clean = pd.DataFrame({
    "train_id": [f"TR-{1000+i}" for i in range(100)],
    "departure_time": [f"{random.randint(0,23):02}:{random.randint(0,59):02}" for _ in range(100)],
    "arrival_time": [f"{random.randint(0,23):02}:{random.randint(0,59):02}" for _ in range(100)],
    "station": [fake.city() for _ in range(100)],
    "platform": [str(random.randint(1, 20)) for _ in range(100)]
})

schedule_dirty = inject_errors(
    schedule_clean,
    {
        "train_id": lambda: random.choice(["###", "TRAIN???", "BAD_ID"]),
        "departure_time": lambda: random.choice(["25:99", "noon", "abc"]),
        "arrival_time": lambda: random.choice(["99:99", "midnightish", "--:--"]),
        "station": lambda: random.choice(["UnknownStation", "123", "###"]),
        "platform": lambda: random.choice(["-1", "PlatformX", "999"])
    }
)

save_dataset("train_schedule", schedule_clean, schedule_dirty)


# ============================================================
# Dataset 7: Product Codes
# SDC Type: Pattern (alphanumeric regex)
# ============================================================

products_clean = pd.DataFrame({
    "product_code": [f"PRD-{random.randint(1000,9999)}" for _ in range(100)],
    "product_name": [fake.word().capitalize() for _ in range(100)],
    "category": [random.choice(["Electronics", "Clothing", "Food", "Books"]) for _ in range(100)],
    "price": [str(round(random.uniform(5, 500), 2)) for _ in range(100)],
    "supplier": [fake.company() for _ in range(100)]
})

products_dirty = inject_errors(
    products_clean,
    {
        "product_code": lambda: random.choice(["###", "INVALID CODE", "123"]),
        "product_name": lambda: random.choice(["@@@@", "", "12345"]),
        "category": lambda: random.choice(["MagicItems", "Unknown", "???"]),
        "price": lambda: random.choice(["-50", "Free", "999999"]),
        "supplier": lambda: random.choice(["###", "NoSupplier", "123"])
    }
)

save_dataset("products", products_clean, products_dirty)


# ============================================================
# Dataset 8: Cities
# SDC Type: CTA (city classifier)
# ============================================================

cities = [
    "Berlin", "Paris", "Tokyo", "Toronto", "Sydney",
    "Madrid", "Rome", "New York"
]

cities_clean = pd.DataFrame({
    "city": [random.choice(cities) for _ in range(100)],
    "country": [random.choice(countries) for _ in range(100)],
    "population": [str(random.randint(100000, 20000000)) for _ in range(100)],
    "timezone": [random.choice(["UTC+1", "UTC+9", "UTC-5", "UTC+10"]) for _ in range(100)],
    "mayor": [fake.name() for _ in range(100)]
})

cities_dirty = inject_errors(
    cities_clean,
    {
        "city": lambda: random.choice(["Laptop", "Blueberry", "InfiniteLoop"]),
        "country": lambda: random.choice(["UnknownLand", "Mars", "Kitchen"]),
        "population": lambda: random.choice(["-500", "ManyPeople", "9999999999"]),
        "timezone": lambda: random.choice(["MoonTime", "UTC+99", "???"]),
        "mayor": lambda: random.choice(["MayorBot", "12345", "@@@@"])
    }
)

save_dataset("cities", cities_clean, cities_dirty)


# ============================================================
# Dataset 9: Days of Week
# SDC Type: CTA (day classifier)
# ============================================================

weekdays = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]

calendar_clean = pd.DataFrame({
    "day": [random.choice(weekdays) for _ in range(100)],
    "employee": [fake.name() for _ in range(100)],
    "shift": [random.choice(["Morning", "Afternoon", "Night"]) for _ in range(100)],
    "hours_worked": [str(round(random.uniform(4, 12), 1)) for _ in range(100)],
    "department": [random.choice(["Sales", "Operations", "Support"]) for _ in range(100)]
})

calendar_dirty = inject_errors(
    calendar_clean,
    {
        "day": lambda: random.choice(["Funday", "Someday", "Workdayish"]),
        "employee": lambda: random.choice(["1234", "@@@@", "NoEmployee"]),
        "shift": lambda: random.choice(["UltraNight", "RandomShift", "???"]),
        "hours_worked": lambda: random.choice(["-2", "40", "all_day"]),
        "department": lambda: random.choice(["UnknownDept", "BananaOps", "123"])
    }
)

save_dataset("employee_schedule", calendar_clean, calendar_dirty)


# ============================================================
# Dataset 10: States
# SDC Type: CTA (state classifier)
# ============================================================

states = [
    "California", "Texas", "Florida", "New York",
    "Nevada", "Ohio", "Illinois", "Arizona"
]

states_clean = pd.DataFrame({
    "state": [random.choice(states) for _ in range(100)],
    "city": [fake.city() for _ in range(100)],
    "zipcode": [fake.postcode() for _ in range(100)],
    "governor": [fake.name() for _ in range(100)],
    "population": [str(random.randint(500000, 40000000)) for _ in range(100)]
})

states_dirty = inject_errors(
    states_clean,
    {
        "state": lambda: random.choice(["MiddleEarth", "Narnia", "Westworld"]),
        "city": lambda: random.choice(["123City", "NowhereTown", "@@@"]),
        "zipcode": lambda: random.choice(["ABCDE", "12", "ZIP???"]),
        "governor": lambda: random.choice(["GovernorBot", "12345", ""]),
        "population": lambda: random.choice(["-1000", "Millions", "999999999"])
    }
)

save_dataset("us_states", states_clean, states_dirty)


# ============================================================
# Finished
# ============================================================

print("\nAll datasets generated successfully.")
print(f"Datasets saved under: {BASE_DIR}")
