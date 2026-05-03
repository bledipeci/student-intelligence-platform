import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import run_pipeline, get_feature_columns
from src.models import load_models


def classify_risk(predicted_grade, pass_prediction, category):
    # if the model predicts the student will fail, they are high risk
    # if they pass but land in the medium performance bucket, they are medium risk
    # otherwise they are low risk
    if pass_prediction == 0 or predicted_grade < 10:
        return "HIGH"
    elif category == "Medium":
        return "MEDIUM"
    else:
        return "LOW"


def generate_recommendations(row, risk_level):
    # we build a list of recommendations by checking each risk factor one by one
    recommendations = []

    # absences — more than 15 is critical, more than 10 is a warning
    if row["absences"] > 15:
        recommendations.append(f"Critical attendance issue - {int(row['absences'])} absences recorded. Immediate action required.")
    elif row["absences"] > 10:
        recommendations.append(f"Reduce absences - {int(row['absences'])} missed classes is impacting performance.")

    # study time — 1 means less than 2 hours a week which is too low
    if row["studytime"] <= 1:
        recommendations.append("Study time is below 2 hours per week - try to set a daily study routine.")
    elif row["studytime"] == 2 and risk_level == "HIGH":
        recommendations.append("Consider increasing study time beyond 2-5 hours per week given current risk level.")

    # past failures — each failure significantly reduces the chance of passing
    if row["failures"] >= 3:
        recommendations.append(f"History of {int(row['failures'])} failures - speak with an academic advisor about a recovery plan.")
    elif row["failures"] > 0:
        recommendations.append(f"Address past failures ({int(row['failures'])}) - targeted tutoring can help prevent further setbacks.")

    # grade trend — if the grade dropped between period 1 and period 2 that is a warning sign
    if row["G1"] > row["G2"] + 2:
        recommendations.append(f"Significant grade drop from Period 1 ({int(row['G1'])}) to Period 2 ({int(row['G2'])}) - identify what changed.")
    elif row["G1"] > row["G2"]:
        recommendations.append("Grades have declined since Period 1 - review recent coursework and seek help early.")

    # alcohol — the alcohol column is the average of weekday and weekend consumption (1 to 5 scale)
    if row["alcohol"] >= 4:
        recommendations.append("Very high alcohol consumption detected - this is strongly associated with lower academic performance.")
    elif row["alcohol"] >= 3:
        recommendations.append("Alcohol consumption may be affecting focus and study quality.")

    # social life — going out a lot while struggling academically is a red flag
    if row["goout"] >= 4 and risk_level in ("HIGH", "MEDIUM"):
        recommendations.append("High social activity combined with academic struggles - consider rebalancing priorities.")

    # health — self-reported health score below 3 on a 1 to 5 scale
    if row["health"] <= 2:
        recommendations.append("Poor self-reported health may be impacting concentration - speak with a school counsellor.")

    # family relationships — low score means difficult home environment
    if row["famrel"] <= 2:
        recommendations.append("Low family relationship quality - school counselling support may help manage home stress.")

    # school support — high risk students with no support should enrol
    if row["schoolsup"] == 0 and risk_level == "HIGH":
        recommendations.append("No school support enrolled - strongly consider signing up for the extra support programme.")

    # no support at all — neither family nor paid tutoring
    if row["famsup"] == 0 and row["paid"] == 0 and risk_level == "HIGH":
        recommendations.append("No family or paid academic support - look into free tutoring resources available at school.")

    # internet access — no internet at home limits access to study materials
    if row["internet"] == 0:
        recommendations.append("No internet access at home - use school library resources for research and online materials.")

    # travel time — 3 or above means more than 30 minutes each way
    if row["traveltime"] >= 3:
        recommendations.append("Long travel time to school - use commute time for light revision or listening to educational content.")

    # ambition vs reality — wants higher education but is currently failing
    if row["higher"] == 1 and risk_level == "HIGH":
        recommendations.append("You want to pursue higher education but are currently at high risk - focus on reversing this trend now.")

    # romantic relationship — can be a distraction for struggling students
    if row["romantic"] == 1 and risk_level == "HIGH":
        recommendations.append("Being in a relationship during a high-risk academic period can be distracting - ensure it is not affecting study time.")

    # too much free time — if at risk, this time should go towards studying
    if row["freetime"] >= 4 and risk_level in ("HIGH", "MEDIUM"):
        recommendations.append("High amount of free time reported - redirect some of it towards structured study sessions.")

    # low parent education — student may not have academic guidance at home
    if row["parent_edu"] < 2 and risk_level == "HIGH":
        recommendations.append("Limited parental academic background - seek guidance from teachers or a mentor outside the home.")

    # positive message for students who are doing well
    if risk_level == "LOW":
        recommendations.append("Performance is strong - consider helping peers or joining academic clubs to further develop skills.")

    # if no issues were found, give a neutral message
    if not recommendations:
        recommendations.append("No major risk factors detected - keep up the current routine and stay consistent.")

    return recommendations


def generate_report(student_row):
    # load the saved models and run predictions for one student
    lr, log_reg, dt = load_models()

    # convert the row into a one-row dataframe so the model can process it
    features = student_row[get_feature_columns()].to_frame().T

    predicted_grade = round(lr.predict(features)[0], 1)
    pass_prediction = log_reg.predict(features)[0]
    category        = dt.predict(features)[0]
    risk_level      = classify_risk(predicted_grade, pass_prediction, category)
    recommendations = generate_recommendations(student_row, risk_level)

    return {
        "predicted_grade": predicted_grade,
        "pass_fail":        "Pass" if pass_prediction == 1 else "Fail",
        "category":         category,
        "risk_level":       risk_level,
        "recommendations":  recommendations,
    }


if __name__ == "__main__":
    df = run_pipeline()

    for i in range(3):
        row = df.iloc[i]
        report = generate_report(row)

        print(f"\nStudent #{i + 1}")
        print(f"  Actual G3:        {int(row['G3'])} / 20")
        print(f"  Predicted Grade:  {report['predicted_grade']} / 20")
        print(f"  Pass/Fail:        {report['pass_fail']}")
        print(f"  Category:         {report['category']}")
        print(f"  Risk Level:       {report['risk_level']}")
        print(f"  Recommendations:")
        for rec in report["recommendations"]:
            print(f"    - {rec}")
