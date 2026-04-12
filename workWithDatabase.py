"""
workWithDatabase.py
-------------------
Database connector and email alert module.

Changes from original:
    - Added `setup_schema()` to create the `trajectories` table if it does not
      exist, linking sightings to the existing `Motobike` table via plate number.
    - Added `insert_trajectory_entry()` to persist each camera sighting.
    - `queryPlate()` now accepts an optional `trajectory` list so that the
      violation email includes the vehicle's full cross-camera journey.
    - All original single-camera behaviour is preserved.
"""

import smtplib
from datetime import datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Optional

import mysql.connector

from emailForm import form


class DatabaseConnector:
    """
    Manages MySQL interactions and email notifications for the traffic
    violation detection system.
    """

    def __init__(self) -> None:
        self.conn = mysql.connector.connect(
            host     = "localhost",
            user     = "root",
            password = None,
            database = "VietNamVehicle",
        )
        self.plateNumber = ""

        # SMTP configuration
        self.smtp_server   = "smtp.office365.com"
        self.smtp_port     = 587
        self.smtp_username = "hoangtrhien203@gmail.com"
        self.smtp_password = "22022003Hth$$"

    # ── Schema management ─────────────────────────────────────────────────────

    def setup_schema(self) -> None:
        """
        Create the `trajectories` table if it does not already exist.

        Schema:
            id              — auto-increment primary key
            global_id       — persistent cross-camera vehicle identity
            camera_id       — camera that recorded this sighting
            local_track_id  — SORT/BoT-SORT track ID within that camera
            entry_time      — ISO-8601 timestamp when vehicle entered view
            exit_time       — ISO-8601 timestamp when vehicle left view
            plate_text      — best OCR plate string for this sighting
            plate_confidence— OCR confidence score [0.0, 1.0]
            is_violation    — whether a violation was detected in this sighting
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trajectories (
                id               INT AUTO_INCREMENT PRIMARY KEY,
                global_id        INT          NOT NULL,
                camera_id        VARCHAR(64)  NOT NULL,
                local_track_id   INT          NOT NULL,
                entry_time       DATETIME     NOT NULL,
                exit_time        DATETIME,
                plate_text       VARCHAR(32)  DEFAULT '',
                plate_confidence FLOAT        DEFAULT 0.0,
                is_violation     TINYINT(1)   DEFAULT 0,
                created_at       TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_global_id (global_id),
                INDEX idx_plate     (plate_text)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        self.conn.commit()
        cursor.close()
        print("[DB] Schema ready — trajectories table OK.")

    # ── Trajectory persistence ────────────────────────────────────────────────

    def insert_trajectory_entry(self,
                                 global_id: int,
                                 camera_id: str,
                                 local_track_id: int,
                                 entry_time: float,
                                 exit_time: Optional[float],
                                 plate_text: str,
                                 plate_confidence: float,
                                 is_violation: bool = False) -> None:
        """
        Insert one camera sighting into the `trajectories` table.

        Parameters
        ----------
        global_id        : persistent cross-camera vehicle ID
        camera_id        : e.g. "cam_0"
        local_track_id   : SORT/BoT-SORT local ID
        entry_time       : Unix timestamp (float)
        exit_time        : Unix timestamp or None if still active
        plate_text       : OCR plate string
        plate_confidence : float in [0, 1]
        is_violation     : True if a violation was detected in this sighting
        """
        def _ts(t):
            if t is None:
                return None
            return datetime.utcfromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO trajectories
                (global_id, camera_id, local_track_id,
                 entry_time, exit_time,
                 plate_text, plate_confidence, is_violation)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (global_id, camera_id, local_track_id,
             _ts(entry_time), _ts(exit_time),
             plate_text, plate_confidence, int(is_violation)),
        )
        self.conn.commit()
        cursor.close()

    def get_trajectory(self, global_id: int) -> List[dict]:
        """
        Retrieve all camera sightings for a given global vehicle ID.

        Returns a list of dicts ordered by entry_time ascending.
        """
        cursor = self.conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT camera_id, local_track_id,
                   entry_time, exit_time,
                   plate_text, plate_confidence, is_violation
            FROM   trajectories
            WHERE  global_id = %s
            ORDER  BY entry_time ASC
            """,
            (global_id,),
        )
        rows = cursor.fetchall()
        cursor.close()
        return rows

    # ── Violation query & email ───────────────────────────────────────────────

    def queryPlate(self,
                   plateFileName: str,
                   trajectory: Optional[List[dict]] = None) -> None:
        """
        Look up the vehicle owner by plate number and send a violation email.

        Parameters
        ----------
        plateFileName : str
            Path to a text file containing the plate number string.
        trajectory : list of dict, optional
            Cross-camera trajectory entries to include in the email body.
            Each dict should have keys: camera_id, entry_time, exit_time,
            plate_text, is_violation.
        """
        with open(plateFileName, "r") as f:
            self.plateNumber = f.read().strip()

        if not self.plateNumber:
            return

        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM Motobike WHERE BienSoXe = %s",
            (self.plateNumber,),
        )
        rows = cursor.fetchall()
        self.conn.commit()

        if not rows:
            cursor.close()
            return

        owner_row = rows[0]
        cursor.close()

        # ── Build trajectory HTML block ───────────────────────────────────────
        trajectory_html = ""
        if trajectory:
            rows_html = "".join(
                f"<tr>"
                f"<td>{e.get('camera_id','')}</td>"
                f"<td>{e.get('entry_time','')}</td>"
                f"<td>{e.get('exit_time','')}</td>"
                f"<td>{'&#10003; Violation' if e.get('is_violation') else 'Normal'}</td>"
                f"</tr>"
                for e in trajectory
            )
            trajectory_html = f"""
            <h3>Vehicle Trajectory</h3>
            <table border="1" cellpadding="6" cellspacing="0">
              <thead>
                <tr>
                  <th>Camera</th><th>Entry Time</th>
                  <th>Exit Time</th><th>Status</th>
                </tr>
              </thead>
              <tbody>{rows_html}</tbody>
            </table>
            """

        # ── Compose email ─────────────────────────────────────────────────────
        html = form.replace("&biensoxe", owner_row[0])
        html = html.replace("&chuxe",    owner_row[3])
        # Inject trajectory table before the closing </body> tag
        if trajectory_html:
            html = html.replace("</body>", trajectory_html + "</body>")

        message = MIMEMultipart("related")
        message["From"]    = self.smtp_username
        message["To"]      = owner_row[10]
        message["Subject"] = "Email thông báo vi phạm"
        message.attach(MIMEText(html, "html", "utf-8"))

        image_path = "./VehicleImageData/" + owner_row[9]
        try:
            with open(image_path, "rb") as attachment:
                image_part = MIMEImage(attachment.read(), name="Anh-Vi-Pham.jpg")
                message.attach(image_part)
        except FileNotFoundError:
            pass   # image not available; send email without it

        server = smtplib.SMTP(self.smtp_server, self.smtp_port)
        server.starttls()
        server.login(self.smtp_username, self.smtp_password)
        server.sendmail(self.smtp_username, owner_row[10], message.as_string())
        server.quit()
        print(f"[DB] Violation email sent to {owner_row[10]} for plate {self.plateNumber}")
