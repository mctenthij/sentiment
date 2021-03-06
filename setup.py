from setuptools import setup

setup(name='sentiment',
      version='0.1',
      description='Sentiment analysis tools',
      author='Marijn ten Thij',
      author_email='mctenthij@gmail.com',
      license='CC-BY-4.0',
      packages=['sentiment'],
      include_package_data=True,
      package_dir={'sentiment': 'sentiment'},
      package_data={'sentiment': ['data/*.txt', 'data/NRC_translations/*.txt', 'data/*.csv']},
      install_requires=[
            'pandas',
            'afinn',
            'nltk',
            'textblob',
            'vaderSentiment'
      ]
)
