import praw

reddit = praw.Reddit(
    client_id="AEJVauFgASesRFuip_1VoQ",
    client_secret="eC26Ck0q9FyuAbE_yS1JYGQH9NrEjQ",
    user_agent="reddit_pipeline_app"
)

subreddit = reddit.subreddit("python")
for post in subreddit.hot(limit=5):
    print(post.title)
    