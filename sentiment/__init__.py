from .sentiment import Sentiment


if __name__ == '__main__':
    s = Sentiment()
    results = s.calculate_average_score("This is a good example to show sentiment analysis!")
    for r in results:
        print("Sentiment score for measure `{r}' is {s:.4f}".format(r=r, s=results[r]))
