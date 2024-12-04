# In js we use Date.now() to get the current time in milliseconds
# In python we use time.time() to get the current time in seconds
# So we need to convert the seconds to milliseconds
unix_seconds_to_ms = lambda seconds: seconds * 1000