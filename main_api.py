import json, uuid, os
from flask import Flask, render_template, jsonify, request, send_file
from SimilarityAnalyser import SimilarityAnalyser
from flask_cors import CORS

app = Flask(__name__, template_folder='static')

CORS(app)

theme_colors = {
    "primary": "#FFC107",
    "secondary": "#F7FFF7",
    "tertiary": "#2F3061",
    "tertiary2": "#6CA6C1",
    "text": "#343434"
}
@app.route('/')
def index():
    return render_template('index.html', colors=theme_colors)


@app.route('/compare', methods=['POST'])
def compare():
    data = json.loads(request.data)
    sp = float(request.args.get('sp'))
    dp = float(request.args.get('dp'))

    similarity_analyzer = SimilarityAnalyser(data['text_1'], data['text_2'], sp, dp)
    stat1 = (float(similarity_analyzer.average_distance_all_cases()),
             "After computing the distance between every pair of sentences, returns the average.",
             "Average distance")
    stat2 = (float(similarity_analyzer.average_distance_after_greedy_assignation()[0]),
             "Computes the similarity scores between all posible sentences in "
             "the 2 pieces of text. Based on these scores, it maches the pair(one "
             "from the first text, the second one from the second text) with the "
             "best score and removes them from the pool. This process repeats "
             "until the maximum number of matches is achieved. Returns the average of these assignations",
             "Average distance after greedy assignation")
    stat4 = (float(similarity_analyzer.get_text_similarity()),
             "Minimum distance: Computes the average between the best matches "
             "of the first sentence in the second and the best matches of the "
             "second sentence in the first one. size penalty : penalizes "
             "sentences of different word counts (proportionate to the "
             "size of the shortest sentence), displacement penalty: "
             "penalizes sentences with different ordering of words "
             "based on the distance between the indexes of words inside the sentences.",
             "Text Similarity")
    # stat3 = int(similarity_analyzer.closest_to_similarity_score(0.4)
    additional_stats = [stat1, stat2, stat4]

    matrix, prop_text_1, prop_text_2 = similarity_analyzer.get_semantic_distance_matrix_heatmap(data['text_1'],
                                                                                                data['text_2'])
    prop_text_1 = [' '.join(prop) for prop in prop_text_1]
    prop_text_2 = [' '.join(prop) for prop in prop_text_2]

    return render_template("report.html", colors=theme_colors, additional_stats=additional_stats, matrix=matrix,
                           prop_text_1=prop_text_1,
                           prop_text_2=prop_text_2)


@app.route('/template')
def template():
    return render_template('report.html')


@app.route('/Chart.HeatMap-0.0.1-alpha/dst/Chart.HeatMap.S.js')
def chartlib():
    return open("./Chart.HeatMap-0.0.1-alpha/dst/Chart.HeatMap.S.js", "r").read()


if __name__ == "__main__":
    app.run(debug=True)
