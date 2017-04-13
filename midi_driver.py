#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Midi-based input control for drawbot_gui, zolaroid, etc

Author: tennessee
Created on: 2017-04-01

Copyright 2017, Tennessee Carmel-Veilleux. All rights reserved.
"""

from __future__ import print_function
import sys
import time
import mido
import threading
import Queue


def test_midi(portname):
    try:
        with mido.open_input(portname) as port:
            print('Using {}'.format(port))
            print('Waiting for messages...')
            for message in port:
                print('Received {}'.format(message))
                sys.stdout.flush()
    except KeyboardInterrupt:
        pass


class MidiControllerDriver(object):
    """
    Uses Mido to read events
    """
    def __init__(self, port_name, channel, event_handler, axes_configs):
        """
        Start a robot HMI driver.

        Axes are configured with following dict:
        { "name": "name_of_axis_here_used_in_events",
          "centered": True|False, # If true, a "center" value will be calculated and removed from each reading}

        :param port_name: MIDI device name to use, None to use first one
        :param channel: Midi channel to parse
        :param event_handler: Event handler that receives the events
        :param axes_configs: List of axis configuration dicts. See above
        """
        self.port_name = port_name
        self.channel = channel
        self.event_handler = event_handler
        self.num_axes = len(axes_configs)
        self.axes_configs = axes_configs
        self.last_values = { control: -1 for control in self.axes_configs.keys() }
        self.event_queue = Queue.Queue()
        self.midi_dev = mido.open_input(self.port_name, callback=(lambda event: self.event_queue.put({"event": event, "ts":time.time()})))
        self.running = True

        self.thread = threading.Thread(target=self.process, name="midi_robot_driver.%s" % str(port_name))
        self.thread.daemon = True
        self.thread.start()

    def process(self):
        while self.running:
            event = self.event_queue.get(block=True)
            if event is False:
                self.running = False
                self.midi_dev.close()
                continue

            # Got an event, only handle control_changes on correct channel for now
            midi_ev = event["event"]
            if midi_ev.type != "control_change" or midi_ev.channel != self.channel:
                continue

            # Disregard events for controls we don't know
            if midi_ev.control not in self.axes_configs:
                continue

            # Parse event, as it is destined to us by now
            ts = event["ts"]

            control = midi_ev.control
            value = midi_ev.value
            axis_config = self.axes_configs[control]

            # Only emit if value is new
            if value != self.last_values[control]:
                self.last_values[control] = value
                if axis_config["centered"]:
                    event_value = (float(value) - 63.0) / 64.0
                else:
                    event_value = float(value) / 127.0

                event = {"event": "control_change", "ts": ts, "control": control,"axis": axis_config["name"], "value": event_value}

                # Button-down-only events are only sent on down
                if axis_config.get("button_down_only") and value != 127:
                    continue

                if self.event_handler is not None:
                    self.event_handler(event)

    def shutdown(self):
        self.event_queue.put(False)
        self.thread.join(1.0)

if __name__ == '__main__':
    def event_handler(event):
        print(event)

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_midi(None if len(sys.argv) < 3 else sys.argv[2])
        sys.exit(0)

    driver = None
    try:
        midi_port = sys.argv[1] if len(sys.argv) > 1 else None
        axes_configs = {
            16:{"name": "axis0", "centered": False},
            0:{"name": "axis1", "centered": True},
            42:{"name": "button_play", "centered": False, "button_down_only": True},
            41:{"name": "button_stop", "centered": False, "button_down_only": True}
        }

        driver = MidiControllerDriver(midi_port, 0, event_handler, axes_configs)
        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        pass
    finally:
        if driver is not None:
            driver.shutdown()