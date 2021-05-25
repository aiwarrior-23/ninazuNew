from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet


class StoreSymptoms(Action):

    def name(self) -> Text:
        return "action_symptoms"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        symptoms = []
        while True:
            sym = tracker.latest_message.get("text")
            if sym == "stop":
                dispatcher.utter_message(text="Thanks for entering your symptoms")
                break
            else:
                symptoms.append(sym)
                continue

        return [SlotSet('allSymptoms', symptoms)]

class PredictDisease(Action):

    def name(self) -> Text:
        return "action_utter_predictions"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        symptoms = list(tracker.get_slot("allSymptoms"))
        dispatcher.utter_message(symptoms)
        return []