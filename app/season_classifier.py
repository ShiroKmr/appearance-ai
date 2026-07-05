import cv2


def classifySeason(analysisResult):
    irisLabel = analysisResult["iris"]["label"]
    skinLabel = analysisResult["skinUndertone"]["label"]
    eyebrowDarkness = analysisResult["eyebrows"]["darkness"]
    eyebrowDensity = analysisResult["eyebrows"]["density"]
    lipLabel = analysisResult["lips"]["label"]

    season = "Unknown"
    reason = []

    isWarm = skinLabel == "Warm"
    isCool = skinLabel == "Cool"
    isNeutral = skinLabel == "Neutral"

    isLightEyes = irisLabel in ["Blue", "Green", "Gray"]
    isDarkEyes = irisLabel in ["Brown", "Dark brown", "Deep dark brown"]
    isSoftEyes = irisLabel in ["Gray", "Hazel", "Green"]

    isDarkBrows = eyebrowDarkness == "Dark"
    isMediumBrows = eyebrowDarkness == "Medium"
    isLightBrows = eyebrowDarkness == "Light"

    isBrightLips = lipLabel in ["Red / rosy", "Pink"]
    isMutedLips = lipLabel in ["Neutral", "Pale", "Peach / brownish"]

    # Seasons need to be chosen more specifically
    if isCool or isNeutral:
        if isDarkEyes and isDarkBrows and isBrightLips:
            season = "Deep Winter"
            reason.append("Cool/neutral skin and high contrast")
        elif (isLightEyes or isSoftEyes) and (isMediumBrows or isDarkBrows):
            season = "Bright Winter"
            reason.append("Cool/neutral skin and medium contrast")
        elif (isLightEyes or isSoftEyes) and isLightBrows and isMutedLips:
            season = "Soft Summer"
            reason.append("Cool/neutral skin and low contrast")
    elif isNeutral:
        if isMutedLips and isSoftEyes:
            season = "Soft Autumn"
            reason.append("Neutral skin and low contrast")
        elif isBrightLips and isLightEyes and isDarkBrows:
            season = "Deep Autumn"
            reason.append("Neutral skin and high contrast")
    elif isCool and isBrightLips:
        if (isDarkEyes or isLightEyes) and isDarkBrows:
            season = "True Winter"
            reason.append("Cool skin and high contrast")
    elif isCool:
        if isLightEyes and isMutedLips and isMediumBrows:
            season = "Light Summer"
            reason.append("Cool skin and medium contrast")
        elif isLightBrows:
            season = "True Summer"
            reason.append("Cool skin and low contrast")
    elif isWarm:
        if isLightEyes and isBrightLips:
            season = "Light Spring"
            reason.append("Warm skin and bright looks")
        elif isLightEyes and isMediumBrows and isBrightLips:
            season = "True Spring"
            reason.append("Warm skin and high contrast")
        elif (isLightEyes or isSoftEyes) and (isMediumBrows or isDarkBrows):
            season = "Bright Spring"
            reason.append("Warm skin and medium contrast")
        else:
            season = "True Autumn"
            reason.append("Warm skin and high contrast")

        
    

    return {
        "season": season,
        "reason": ", ".join(reason),
        "features": {
            "iris": irisLabel,
            "skinUndertone": skinLabel,
            "eyebrowDarkness": eyebrowDarkness,
            "eyebrowDensity": eyebrowDensity,
            "lips": lipLabel,
        }
    }


def drawSeasonResult(frame, seasonResult):
    season = seasonResult["season"]

    cv2.putText(
        frame,
        f"Season: {season}",
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        seasonResult["reason"],
        (20, 215),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )