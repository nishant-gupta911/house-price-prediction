#!/usr/bin/env python3
"""
Test script for map functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test imports
try:
    import folium
    print("✅ Folium imported successfully")
except ImportError as e:
    print(f"❌ Failed to import folium: {e}")

try:
    from streamlit_folium import st_folium
    print("✅ Streamlit-folium imported successfully")
except ImportError as e:
    print(f"❌ Failed to import streamlit-folium: {e}")

try:
    from geopy.geocoders import Nominatim
    print("✅ Geopy imported successfully")
except ImportError as e:
    print(f"❌ Failed to import geopy: {e}")

# Test basic map creation
try:
    print("\n🗺️ Testing basic map creation...")
    
    # Sample coordinates (Ames, Iowa)
    lat, lng = 42.030, -93.630
    predicted_price = 250000
    
    # Create a simple map
    test_map = folium.Map(
        location=[lat, lng],
        zoom_start=15,
        tiles='OpenStreetMap'
    )
    
    # Add a marker
    folium.Marker(
        location=[lat, lng],
        popup=f"🏠 Test House<br>💰 ${predicted_price:,.0f}",
        tooltip="Test House Location",
        icon=folium.Icon(color='red', icon='home', prefix='fa')
    ).add_to(test_map)
    
    print("✅ Basic map created successfully")
    
    # Test geocoding
    print("\n📍 Testing geocoding...")
    geolocator = Nominatim(user_agent="house_price_test")
    location = geolocator.geocode("Ames, Iowa")
    
    if location:
        print(f"✅ Geocoding successful: {location.latitude}, {location.longitude}")
        print(f"   Address: {location.address}")
    else:
        print("❌ Geocoding failed")
        
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()

print("\n🎯 Map functionality test completed!")
