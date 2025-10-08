import praw

reddit = praw.Reddit(
    client_id="YOUR CLIENT ID",
    client_secret="YOUR CLIENT SECRET",
    user_agent="reddit_pipeline_app"
)

subreddit = reddit.subreddit("python")
for post in subreddit.hot(limit=5):
    print(post.title)
    
