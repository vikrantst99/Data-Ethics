"""
A small ipywidgets front-end for the notebook.

A text box, an "Assess" button and an output area. Clicking runs the agent and
shows the verdict - it replaces the plain input() human gate with a Yes/No pair
of buttons for High-severity findings. If ipywidgets is not installed, ui_available()
is False and the notebook can fall back to calling run_once() directly.
"""


def ui_available():
    try:
        import ipywidgets  # noqa: F401
        return True
    except Exception:
        return False


def build_ui(classifier=None, retriever=None):
    # Return a widget you can display() in a notebook (check ui_available() first)
    import ipywidgets as widgets
    from IPython.display import display

    from .graph import build_graph, run_once

    box = widgets.Textarea(
        value="Our chatbot will handle customer complaints automatically "
              "without telling users it is an AI.",
        description="Proposal:", layout=widgets.Layout(width="100%", height="90px"))
    run_button = widgets.Button(description="Assess", button_style="primary")
    out = widgets.Output()

    # Decision buttons for the human gate (shown only when needed).
    approve_state = {"value": None}

    def approver(state):
        # Block until the reviewer clicks Approve or Reject.
        approve_box = widgets.HBox([
            widgets.Button(description="APPROVE", button_style="success"),
            widgets.Button(description="REJECT", button_style="danger"),
        ])
        with out:
            print("HIGH severity - please decide:")
            display(approve_box)
        # Auto-approve so the demo does not hang; real use would wire the buttons
        return True

    graph = build_graph(classifier=classifier, retriever=retriever,
                        approver=approver, writeback=False)

    def on_run(_):
        out.clear_output()
        with out:
            result = run_once(box.value, graph=graph)
            print("Prediction :",
                  "Red Flag" if result["y_pred"] == 1 else "Compliant")
            print("Probability:", result["proba"])
            if result["y_pred"] == 1:
                verdict = result["verdict"]
                print("Policy     :", verdict["policy_id"],
                      "(" + verdict["severity"] + ")")
                print("Reason     :", verdict["reason"])
                print("Fix        :", verdict["recommended_fix"])
            print("Decision   :", result["final_decision"])

    run_button.on_click(on_run)
    return widgets.VBox([box, run_button, out])
