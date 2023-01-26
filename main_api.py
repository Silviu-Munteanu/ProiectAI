import json, uuid, os
from flask import Flask, render_template, jsonify, request, send_file
from SimilarityAnalyser import SimilarityAnalyser
from flask_cors import CORS

app = Flask(__name__, template_folder='static')

CORS(app)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/compare', methods=['POST'])
def compare():
    data = json.loads(request.data)
    sp = float(request.args.get('sp'))
    dp = float(request.args.get('dp'))

    similarity_analyzer = SimilarityAnalyser(data['text_1'], data['text_2'], sp, dp)
    stat1 = int(similarity_analyzer.average_distance_all_cases())
    stat2 = int(similarity_analyzer.average_distance_after_greedy_assignation()[0])
    # stat3 = int(similarity_analyzer.closest_to_similarity_score(0.4)
    stat4 = int(similarity_analyzer.get_text_similarity())

    matrix, prop_text_1, prop_text_2 = similarity_analyzer.get_semantic_distance_matrix_heatmap(data['text_1'],
                                                                                                data['text_2'])
    prop_text_1 = [' '.join(prop) for prop in prop_text_1]
    prop_text_2 = [' '.join(prop) for prop in prop_text_2]
    
    return render_template("report.html", stat1=stat1, stat2=stat2, stat4=stat4, matrix=matrix, prop_text_1=prop_text_1,
                           prop_text_2=prop_text_2)


@app.route('/template')
def template():
    return render_template('report.html')


@app.route('/Chart.HeatMap-0.0.1-alpha/dst/Chart.HeatMap.S.js')
def chartlib():
    return open("./Chart.HeatMap-0.0.1-alpha/dst/Chart.HeatMap.S.js", "r").read()


if __name__ == "__main__":
    app.run(debug=True)
