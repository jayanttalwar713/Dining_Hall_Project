# dining_hall_predictor_csv_menus.py
# Cornell Dining Hall Busyness Predictor - With CSV Menu Integration
# Uses local CSV file for menu data instead of API calls

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import requests

# =============================================================================
# CONFIGURATION - CUSTOMIZE THESE FOR YOUR CAMPUS
# =============================================================================

# Updated with your Cornell dining halls
DINING_HALLS = ["Risley", "Okenshields", "Appel", "Becker", "Cook", "Bethe", "Keeton", "Morrison", "Rose", "Trillium", "Terrace", "Macs"]
MEALS = ["Breakfast", "Lunch", "Dinner"]

# Your actual operating hours - if a hall doesn't offer a meal, just don't include that key!
OPERATING_HOURS = {
    "Risley": {
        "Lunch": (11.0, 14.0),      # 11:00am - 2:00pm
        "Dinner": (17.0, 19.0)      # 5:00pm - 7:00pm
    },
    "Okenshields": {
        "Lunch": (10.5, 14.5),      # 10:30am - 2:30pm
        "Dinner": (16.5, 21.0)      # 4:30pm - 9:00pm
    },
    "Appel": {
        "Dinner": (17.5, 20.5)      # 5:30pm - 8:30pm
    },
    "Becker": {
        "Breakfast": (7.0, 10.5),   # 7:00am - 10:30am
        "Lunch": (10.5, 14.0),      # 10:30am - 2:00pm
        "Dinner": (17.0, 20.0)      # 5:00pm - 8:00pm
    },
    "Cook": {
        "Breakfast": (7.0, 10.0),   # 7:00am - 10:00am
        "Dinner": (17.5, 21.0)      # 5:30pm - 9:00pm
    },
    "Bethe": {
        "Breakfast": (7.0, 10.5),   # 7:00am - 10:30am
        "Lunch": (10.5, 14.0),      # 10:30am - 2:00pm
        "Dinner": (16.5, 19.5)      # 4:30pm - 7:30pm
    },
    "Keeton": {
        "Breakfast": (7.0, 10.0),   # 7:00am - 10:00am
        "Dinner": (17.0, 20.0)      # 5:00pm - 8:00pm
    },
    "Morrison": {
        "Breakfast": (7.0, 10.5),   # 7:00am - 10:30am
        "Lunch": (10.5, 14.0),      # 10:30am - 2:00pm
        "Dinner": (17.0, 20.5)      # 5:00pm - 8:30pm
    },
    "Rose": {
        "Breakfast": (7.0, 10.0),   # 7:00am - 10:00am
        "Dinner": (17.0, 20.0)      # 5:00pm - 8:00pm
    },
    "Macs": {
        "Breakfast": (8.0, 19.5),
        "Lunch": (8.0, 19.5),       # 7:00am - 10:00am
        "Dinner": (8.0, 19.5)       # 5:00pm - 8:00pm
    },
    "Terrace": {
        "Lunch": (10.0, 15.0)       # 7:00am - 10:00am
    },
    "Trillium": {
        "Breakfast": (8.0, 15.0),   # 7:00am - 10:00am
        "Lunch": (8.0, 15.0),       # 5:00pm - 8:00pm
    }
}

# Your popularity scores - adjust based on observations
HALL_POPULARITY = {
    "Risley": 0.05,
    "Okenshields": 0.19,  # Most popular - central location
    "Appel": 0.15,
    "Becker": 0.14,
    "Cook": 0.14,
    "Bethe": 0.14,
    "Keeton": 0.14,
    "Morrison": 0.15,
    "Rose": 0.14,
    "Macs": 0.17,
    "Terrace": 0.16,
    "Trillium": 0.01
}

# Your rush hour patterns
RUSH_PATTERNS = {
    "Breakfast": {
        7: 0.1, 8: 0.5, 9: 0.7, 10: 0.5
    },
    "Lunch": {
        11: 0.4, 12: 0.8, 13: 0.9, 14: 0.5
    },
    "Dinner": {
        17: 0.5, 18: 0.9, 19: 0.8, 20: 0.5, 21: 0.1
    },
}

# =============================================================================
# MENU SCORING SYSTEM
# =============================================================================

# Keywords that indicate popular menu items (adjust based on observations)
POPULAR_MENU_KEYWORDS = {
    "stir fry": 0.15,       # Adds 15% busyness
    "pasta bar": 0.12,
    "chicken wings": 0.10,
    "pizza": 0.08,
    "taco": 0.10,
    "korean": 0.15,
    "sushi": 0.12,
    "make your own": 0.10,
    "bbq": 0.12,
    "noodle": 0.10,
    "burrito": 0.08,
    "waffle": 0.07,
    "pancake": 0.06,
    "burger": 0.08,
    "fried chicken": 0.12,
    "mac and cheese": 0.09,
    "special": 0.08,
}

# =============================================================================
# CSV MENU LOADING
# =============================================================================

MENU_CSV_FILE = "dining_menus.csv"

def load_menus_from_csv():
    """
    Loads menu data from CSV file
    
    Expected CSV format:
    date,hall,meal,item
    2026-02-10,Okenshields,Lunch,Pizza
    2026-02-10,Okenshields,Lunch,Pasta Bar
    
    Returns:
        dict of {(hall, date_str, meal): [menu_items]}
    """
    try:
        df = pd.read_csv(MENU_CSV_FILE)
        
        # Validate required columns
        required_cols = ['date', 'hall', 'meal', 'item']
        if not all(col in df.columns for col in required_cols):
            st.error(f"❌ Menu CSV must have columns: {required_cols}")
            return {}
        
        # Convert date to string format for consistent lookup
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        
        # Group by (hall, date, meal) and collect items into lists
        menu_schedule = {}
        grouped = df.groupby(['hall', 'date', 'meal'])
        
        for (hall, date, meal), group in grouped:
            items = group['item'].tolist()
            menu_schedule[(hall, date, meal)] = items
        
        return menu_schedule
        
    except FileNotFoundError:
        st.warning(f"⚠️ Menu file '{MENU_CSV_FILE}' not found - running without menu data")
        return {}
    except Exception as e:
        st.warning(f"⚠️ Error loading menu file: {str(e)}")
        return {}

def calculate_menu_boost(hall, dt, meal, menu_schedule):
    """
    Calculate busyness boost based on menu items
    
    menu_schedule: dict from load_menus_from_csv()
    """
    date_str = dt.strftime("%Y-%m-%d")
    menu_key = (hall, date_str, meal)
    
    if menu_key not in menu_schedule:
        return 0.0, []
    
    items = menu_schedule[menu_key]
    boost = 0.0
    matched_items = []
    
    for item in items:
        item_lower = item.lower()
        for keyword, value in POPULAR_MENU_KEYWORDS.items():
            if keyword in item_lower:
                boost += value
                if item not in matched_items:
                    matched_items.append(item)
                break  # Only count each item once
    
    # Cap at 25%
    final_boost = min(boost, 0.25)
    return final_boost, matched_items

# Academic calendar file
ACADEMIC_CALENDAR_FILE = "academic_calendar.csv"

# Grace period (minutes) - ONLY APPLIES TO CLOSING TIME
GRACE_PERIOD_MINUTES = 30

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def detect_meal_from_time(dt):
    """Auto-detect meal period based on time"""
    hour = dt.hour + dt.minute / 60.0
    if 7.0 <= hour < 11.0:
        return "Breakfast"
    elif 11.0 <= hour < 16.0:
        return "Lunch"
    elif 16.5 <= hour < 22.0:
        return "Dinner"
    else:
        return None

def get_next_opening(hall, current_dt):
    """
    Find the next time this hall opens
    Returns: (meal_name, opening_time) or None
    """
    if hall not in OPERATING_HOURS:
        return None
    
    current_hour = current_dt.hour + current_dt.minute / 60.0
    
    # Check today's remaining meals in order
    for meal in ["Breakfast", "Lunch", "Dinner"]:
        if meal not in OPERATING_HOURS[hall]:
            continue
        start_hour, end_hour = OPERATING_HOURS[hall][meal]
        
        # If this meal hasn't started yet today
        if current_hour < start_hour:
            return (meal, start_hour)
        
        # If we're currently in this meal period
        if start_hour <= current_hour <= end_hour:
            return (meal, start_hour)
    
    # All today's meals have passed - check tomorrow
    for meal in ["Breakfast", "Lunch", "Dinner"]:
        if meal in OPERATING_HOURS[hall]:
            start_hour, _ = OPERATING_HOURS[hall][meal]
            return (f"{meal} (tomorrow)", start_hour)
    
    return None

def is_hall_open(hall, dt, meal):
    """
    Check if a dining hall is open
    Returns: "open", "closed", or "grace"
    Grace period ONLY applies after closing (not before opening)
    """
    if hall not in OPERATING_HOURS:
        return "closed"
    if meal not in OPERATING_HOURS[hall]:
        return "closed"
    
    start_hour, end_hour = OPERATING_HOURS[hall][meal]
    current_hour = dt.hour + dt.minute / 60.0
    
    # Grace period ONLY at the end (closing time)
    grace_end = end_hour + (GRACE_PERIOD_MINUTES / 60.0)
    
    if start_hour <= current_hour <= end_hour:
        return "open"
    elif end_hour < current_hour <= grace_end:
        return "grace"  # Only after closing
    else:
        return "closed"

def format_time_12hr(decimal_hour):
    """Convert decimal hour to 12-hour format"""
    hour = int(decimal_hour)
    minute = int((decimal_hour - hour) * 60)
    period = "AM" if hour < 12 else "PM"
    display_hour = hour if hour <= 12 else hour - 12
    if display_hour == 0:
        display_hour = 12
    return f"{display_hour}:{minute:02d} {period}"

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def generate_features(hall, dt, meal, academic_flags, weather, menu_boost):
    """
    Converts inputs into numerical features
    Total: 30 features (9 halls + other features + menu_boost)
    """
    # === TIME FEATURES ===
    minutes = dt.hour * 60 + dt.minute
    sin_time = np.sin(2 * np.pi * minutes / 1440)
    cos_time = np.cos(2 * np.pi * minutes / 1440)
    
    day = dt.weekday()
    sin_day = np.sin(2 * np.pi * day / 7)
    cos_day = np.cos(2 * np.pi * day / 7)
    is_weekend = 1 if day >= 5 else 0
    hour = dt.hour
    
    # === DINING HALL FEATURES === (9 halls)
    hall_onehot = [1 if hall == h else 0 for h in DINING_HALLS]
    
    # === MEAL FEATURES === (3 meals)
    meal_onehot = [1 if meal == m else 0 for m in MEALS]
    
    # === ACADEMIC CALENDAR FEATURES ===
    acad_flags = [
        academic_flags.get("prelims", 0),
        academic_flags.get("finals", 0),
        academic_flags.get("break", 0)
    ]
    
    # === WEATHER FEATURES ===
    temp = weather.get("temperature", 20)
    precip = weather.get("precipitation", 0)
    is_bad_weather = 1 if (temp < 32 or temp > 85 or precip > 0) else 0
    weather_features = [temp, precip, is_bad_weather]
    
    # === SEASONAL FEATURES ===
    month_sin = np.sin(2 * np.pi * dt.month / 12)
    month_cos = np.cos(2 * np.pi * dt.month / 12)
    
    # === MENU FEATURE ===
    # This boosts predictions when popular items are served
    
    features = np.array([
        sin_time, cos_time,         # 2
        sin_day, cos_day,           # 2
        is_weekend, hour,           # 2
        *hall_onehot,               # 9
        *meal_onehot,               # 3
        *acad_flags,                # 3
        *weather_features,          # 3
        month_sin, month_cos,       # 2
        menu_boost                  # 1
    ])  # Total: 30 features
    
    return features.reshape(1, -1)

# =============================================================================
# HEURISTIC LABELS
# =============================================================================

def generate_busyness_score(hall, dt, meal, academic_flags, weather, menu_boost):
    """
    Generates heuristic busyness score (0 to 1)
    Returns 0 if hall is closed
    """
    status = is_hall_open(hall, dt, meal)
    if status == "closed":
        return 0.0
    elif status == "grace":
        return 0.1
    
    # === BASE SCORE ===
    base = RUSH_PATTERNS.get(meal, {}).get(dt.hour, 0.3)
    
    # === WEEKEND ADJUSTMENT ===
    if dt.weekday() >= 5:
        base *= 0.7
    
    # === ACADEMIC STRESS ===
    acad_factor = (
        0.15 * academic_flags.get("prelims", 0) +
        0.20 * academic_flags.get("finals", 0)
    )
    
    # === BREAK PERIOD ===
    if academic_flags.get("break", 0):
        base *= 0.4
    
    # === WEATHER ===
    temp = weather.get("temperature", 20)
    precip = weather.get("precipitation", 0)
    weather_factor = 0
    if precip > 0 or temp < 32 or temp > 85:
        weather_factor = 0.15
    
    # === HALL POPULARITY ===
    hall_factor = HALL_POPULARITY.get(hall, 0.05)
    
    # === MENU BOOST ===
    # Popular menu items increase predicted busyness
    
    # === COMBINE ALL FACTORS ===
    score = base + acad_factor + weather_factor + hall_factor + menu_boost
    score += np.random.normal(0, 0.05)
    
    return np.clip(score, 0, 1)

# =============================================================================
# WEATHER API
# =============================================================================

def fetch_weather_ithaca():
    """Fetches current weather for Ithaca, NY"""
    url = "https://api.open-meteo.com/v1/forecast?latitude=42.44&longitude=-76.50&current_weather=true&hourly=precipitation&temperature_unit=fahrenheit&timezone=America/New_York"
    try:
        r = requests.get(url, timeout=5)
        data = r.json()
        temp = data["current_weather"]["temperature"]
        current_hour = datetime.now().hour
        precip = 0
        if "hourly" in data and "precipitation" in data["hourly"]:
            precip = data["hourly"]["precipitation"][current_hour]
        return {"temperature": temp, "precipitation": precip}
    except Exception as e:
        return {"temperature": 50, "precipitation": 0}

# =============================================================================
# ACADEMIC CALENDAR
# =============================================================================

def load_academic_calendar():
    """Loads the academic calendar CSV file"""
    try:
        df = pd.read_csv(ACADEMIC_CALENDAR_FILE)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        # Create a basic one if not found
        return pd.DataFrame(columns=['date', 'prelims', 'finals', 'break'])
    except Exception as e:
        st.error(f"Error loading academic calendar: {e}")
        return pd.DataFrame(columns=['date', 'prelims', 'finals', 'break'])

def get_academic_flags(calendar_df, dt):
    """Gets academic flags for a specific datetime"""
    if calendar_df.empty:
        return {"prelims": 0, "finals": 0, "break": 0}
    
    row = calendar_df[calendar_df['date'] == dt.date()]
    if row.empty:
        return {"prelims": 0, "finals": 0, "break": 0}
    
    return row.iloc[0][['prelims', 'finals', 'break']].to_dict()

# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_model():
    """Trains a Gradient Boosting model on synthetic data"""
    st.info("🔄 Training model... This takes ~30 seconds")
    
    calendar_df = load_academic_calendar()
    X, y = [], []
    
    start = datetime.now() - timedelta(days=30)
    meal_times = {
        "Breakfast": range(7, 12),
        "Lunch": range(11, 16),
        "Dinner": range(17, 22),
    }
    
    sample_count = 0
    for day_offset in range(60):
        current_day = start + timedelta(days=day_offset)
        for hall in DINING_HALLS:
            for meal in MEALS:
                for hour in meal_times[meal]:
                    for minute in [0, 15, 30, 45]:
                        dt = current_day.replace(hour=hour % 24, minute=minute)
                        academic_flags = get_academic_flags(calendar_df, dt)
                        weather = {
                            "temperature": 50 + np.random.randn() * 20,
                            "precipitation": max(0, np.random.randn() * 3)
                        }
                        
                        # Random menu boost for training
                        menu_boost = np.random.uniform(0, 0.15)
                        
                        features = generate_features(hall, dt, meal, academic_flags, weather, menu_boost)
                        score = generate_busyness_score(hall, dt, meal, academic_flags, weather, menu_boost)
                        
                        X.append(features.flatten())
                        y.append(score)
                        sample_count += 1
    
    X = np.array(X)
    y = np.array(y)
    
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42
    )
    model.fit(X, y)
    joblib.dump(model, "busyness_model.pkl")
    
    st.success(f"✅ Model trained on {sample_count:,} data points")
    return model

# =============================================================================
# STREAMLIT DASHBOARD
# =============================================================================

def main():
    st.set_page_config(page_title="Dining Hall Predictor", page_icon="🍽️", layout="wide")
    st.title("🍽️ Cornell Dining Hall Busyness Predictor")
    st.caption("Real-time predictions with CSV menu integration")
    
    # === SIDEBAR ===
    with st.sidebar:
        st.header("⚙️ Settings")
        
        if st.button("🔄 Retrain Model", help="Rebuild the prediction model"):
            model = train_model()
            st.rerun()
        
        st.divider()
        
        weather = fetch_weather_ithaca()
        st.metric("🌡️ Temperature", f"{weather['temperature']:.1f}°F")
        st.metric("🌧️ Precipitation", f"{weather['precipitation']:.1f}mm")
        
        st.divider()
        
        with st.expander("📋 Dining Hall Hours"):
            for hall in DINING_HALLS:
                st.write(f"**{hall}**")
                if hall in OPERATING_HOURS:
                    for meal, (start, end) in OPERATING_HOURS[hall].items():
                        st.caption(f"{meal}: {format_time_12hr(start)} - {format_time_12hr(end)}")
                else:
                    st.caption("Hours not configured")
                st.write("")
        
        st.caption("💡 Meal periods auto-detected")
        st.caption("🍜 Menus loaded from CSV file")
        st.caption("⏰ 30-min grace period after closing")
    
    # === LOAD MODEL ===
    try:
        model = joblib.load("busyness_model.pkl")
    except FileNotFoundError:
        st.warning("⚠️ No trained model found. Training now...")
        model = train_model()
    
    # === LOAD MENUS FROM CSV ===
    menu_schedule = load_menus_from_csv()
    if menu_schedule:
        st.success(f"✅ Loaded menus for {len(menu_schedule)} meal periods")
    else:
        st.info("ℹ️ Running without menu data")
    
    # === TIME SLIDER ===
    col1, col2 = st.columns([3, 1])
    
    with col1:
        time_offset = st.slider(
            "⏰ Time from Now",
            min_value=0,
            max_value=120,
            value=0,
            step=15,
            help="Slide to see predictions for future times",
            format="%d min"
        )
    
    with col2:
        if time_offset > 0:
            if st.button("↺ Reset to Now"):
                st.rerun()
    
    selected_dt = datetime.now() + timedelta(minutes=time_offset)
    detected_meal = detect_meal_from_time(selected_dt)
    
    time_str = selected_dt.strftime('%A, %B %d at %I:%M %p')
    if detected_meal:
        st.info(f"📅 **{time_str}** • Meal Period: **{detected_meal}**")
    else:
        st.warning(f"📅 **{time_str}** • ⚠️ Outside normal dining hours")
        st.stop()
    
    # === GET CONTEXT DATA ===
    calendar_df = load_academic_calendar()
    academic_flags = get_academic_flags(calendar_df, selected_dt)
    
    if any(academic_flags.values()):
        active_flags = [k.capitalize() for k, v in academic_flags.items() if v]
        st.warning(f"📚 **Academic Period:** {', '.join(active_flags)}")
    
    # === GENERATE PREDICTIONS ===
    predictions = []
    
    for hall in DINING_HALLS:
        status = is_hall_open(hall, selected_dt, detected_meal)
        
        if status == "closed":
            next_opening = get_next_opening(hall, selected_dt)
            predictions.append((hall, 0.0, "closed", next_opening, [], 0.0))
        else:
            # Hall is open or in grace period - get actual prediction
            menu_boost, matched_items = calculate_menu_boost(hall, selected_dt, detected_meal, menu_schedule)
            features = generate_features(hall, selected_dt, detected_meal, academic_flags, weather, menu_boost)
            score = model.predict(features)[0]
            
            # Ensure score is non-negative, but DON'T override open status!
            # Low score just means "not busy", not "closed"
            score = max(0.0, score)
            
            predictions.append((hall, score, status, None, matched_items, menu_boost))
    
    # Sort: closed at bottom, open by busyness
    predictions.sort(key=lambda x: (x[2] == "closed", x[1]))
    
    # === DISPLAY PREDICTIONS ===
    st.subheader("📊 Dining Hall Rankings")
    
    open_halls = [p for p in predictions if p[2] != "closed"]
    closed_halls = [p for p in predictions if p[2] == "closed"]
    
    if open_halls:
        st.write("**Open Now** (Least → Most Busy)")
        
        for i, prediction in enumerate(open_halls):
            hall = prediction[0]
            score = prediction[1]
            status = prediction[2]
            matched_items = prediction[4]
            menu_boost = prediction[5]
            
            if score < 0.35:
                busy_level = "🟢 Low"
            elif score < 0.65:
                busy_level = "🟡 Medium"
            else:
                busy_level = "🔴 High"
            
            col1, col2, col3, col4 = st.columns([1, 3, 2, 2])
            
            with col1:
                st.write(f"**#{i+1}**")
            
            with col2:
                hall_text = f"**{hall}**"
                if menu_boost > 0.08:
                    hall_text += " 🍜"  # Popular menu indicator
                st.write(hall_text)
            
            with col3:
                st.write(busy_level)
            
            with col4:
                st.progress(score, text=f"{score*100:.0f}%")
            
            # Show menu items if any matched
            if matched_items:
                with st.expander(f"🍽️ Popular items at {hall}", expanded=False):
                    for item in matched_items[:5]:  # Show top 5
                        st.caption(f"• {item}")
    
    if closed_halls:
        st.divider()
        st.write("**Closed**")
        
        for prediction in closed_halls:
            hall = prediction[0]
            next_opening = prediction[3]
            
            col1, col2, col3 = st.columns([1, 3, 2])
            
            with col1:
                st.write("❌")
            
            with col2:
                st.write(f"**{hall}**")
            
            with col3:
                if next_opening:
                    meal_name, open_time = next_opening
                    st.caption(f"Opens at {format_time_12hr(open_time)} ({meal_name})")
                else:
                    st.caption("Check hours")
    
    # === FOOTER ===
    st.divider()
    st.caption("⚠️ Predictions are estimates. Menu data from CSV file.")
    st.caption("🍜 = Popular menu items today • Built with Streamlit")

if __name__ == "__main__":
    main()