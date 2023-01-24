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
    
    similarity_analyzer = SimilarityAnalyser(data['text_1'], data['text_2'])
    stat1 = similarity_analyzer.average_distance_all_cases()
    stat2 = similarity_analyzer.average_distance_after_greedy_assignation()[0]
    stat3 = similarity_analyzer.closest_to_similarity_score(0.4)
    stat4 = similarity_analyzer.get_text_similarity()
    
    figid = str(uuid.uuid1())
    savepath = "./static/figs/" + figid + ".png"
    similarity_analyzer.save_semantic_distance_matrix_heatmap(data['text_1'], data['text_2'], savepath)

    return render_template("report.html", stat1=stat1, stat2=stat2, stat3=stat3, stat4=stat4, fig=figid+".png")

@app.route('/<path:path>')
def fig(path):
    if ".png" in path:
        return send_file(os.path.join("./static/figs", path), mimetype='image/gif')
    

@app.route('/template')
def template():
    return render_template('report.html')


if __name__ == "__main__":
    app.run(debug=True)

