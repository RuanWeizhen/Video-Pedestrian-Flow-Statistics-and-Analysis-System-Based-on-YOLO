from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout


class DashboardSection(QFrame):
    def __init__(self, title: str, subtitle: str | None = None, parent=None):
        super().__init__(parent)
        self.setObjectName("DashboardSection")
        self.setFrameShape(QFrame.StyledPanel)
        self.setProperty("card", True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        header = QVBoxLayout()
        header.setSpacing(2)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("DashboardSectionTitle")
        self.subtitle_label = QLabel(subtitle or "")
        self.subtitle_label.setObjectName("DashboardSectionSubtitle")
        self.subtitle_label.setVisible(bool(subtitle))

        header.addWidget(self.title_label)
        header.addWidget(self.subtitle_label)
        layout.addLayout(header)

        self.body_layout = QVBoxLayout()
        self.body_layout.setSpacing(12)
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self.body_layout)

    def set_body_widget(self, widget):
        self.body_layout.addWidget(widget)

    def set_title(self, title: str):
        self.title_label.setText(str(title))

    def set_subtitle(self, subtitle: str):
        text = str(subtitle or "")
        self.subtitle_label.setText(text)
        self.subtitle_label.setVisible(bool(text))


class DashboardMetricCard(QFrame):
    def __init__(self, title: str, value: str = "0", hint: str = "", accent: str = "#2f80ed", parent=None):
        super().__init__(parent)
        self.setObjectName("DashboardMetricCard")
        self.setProperty("card", True)
        self.setProperty("accent", accent)
        self.setMinimumWidth(100)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(6)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("DashboardMetricTitle")
        self.title_label.setAlignment(Qt.AlignCenter)

        self.value_label = QLabel(value)
        self.value_label.setObjectName("DashboardMetricValue")
        self.value_label.setAlignment(Qt.AlignCenter)
        self.value_label.setWordWrap(True)

        self.hint_label = QLabel(hint)
        self.hint_label.setObjectName("DashboardMetricHint")
        self.hint_label.setAlignment(Qt.AlignCenter)

        layout.addStretch()
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.hint_label)
        layout.addStretch()

    def set_value(self, value: str, hint: str = ""):
        self.value_label.setText(str(value))
        if hint:
            self.hint_label.setText(hint)

    def set_title(self, title: str):
        self.title_label.setText(str(title))

    def set_hint(self, hint: str):
        self.hint_label.setText(str(hint))


class SidebarButton(QPushButton):
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(42)


class UserBadge(QFrame):
    def __init__(self, title: str, value: str, parent=None):
        super().__init__(parent)
        self.setObjectName("UserBadge")
        self.setProperty("card", True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(2)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("UserBadgeTitle")
        self.value_label = QLabel(value)
        self.value_label.setObjectName("UserBadgeValue")
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)

    def set_value(self, value: str):
        self.value_label.setText(str(value))


def build_metric_row(metrics: list[DashboardMetricCard]) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(14)
    for metric in metrics:
        row.addWidget(metric, 1)
    return row
