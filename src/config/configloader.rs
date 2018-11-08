use super::keycode::Input;
use log::warn;
use std::{
    fs::{File, OpenOptions},
    io::{self, SeekFrom, Write},
};
use toml;
use winit::{ElementState, MouseButton, VirtualKeyCode};

gen_config! {
    ConfigControls controls {
        ConfigControlsMovement movement {
            forwards String Input "w".to_uppercase()
            backwards String Input "s".to_uppercase()
            left String Input "a".to_uppercase()
            right String Input "d".to_uppercase()
            down String Input "l_control".to_uppercase()
            up String Input "space".to_uppercase()
            speed_up String Input "scroll_up".to_uppercase()
            speed_down String Input "scroll_down".to_uppercase()
        },
        ConfigControlsSensitivity sensitivity {
            mouse_speed f32 f32 1.0
            movement_speed f32 f32 1.0
            scroll_speed f32 f32 1.0
        },
        ConfigControlsEngine engine {
            grab_cursor String Input "g".to_uppercase()
            wireframe String Input "f".to_uppercase()
        }
    },
    ConfigGraphics graphics {
        ConfigGraphicsSettings settings {
            vsync bool bool true
        }
    }
}

impl Config {
    pub fn new(confpath: &str) -> Self {
        let config = option::new(confpath);

        // TODO: marker for config change
        let config_controls = config.controls.unwrap_or_else(|| {
            warn!("config.controls is invalid, using default");
            Default::default()
        });

        let config_controls_movement = config_controls.movement.unwrap_or_else(|| {
            warn!("config.controls.movement is invalid, using default");
            Default::default()
        });
        let config_controls_sensitivity = config_controls.sensitivity.unwrap_or_else(|| {
            warn!("config.controls.sensitivity is invalid, using default");
            Default::default()
        });
        let config_controls_engine = config_controls.engine.unwrap_or_else(|| {
            warn!("config.controls.engine is invalid, using default");
            Default::default()
        });

        let config_graphics = config.graphics.unwrap_or_else(|| {
            warn!("config.graphics is invalid, using default");
            Default::default()
        });

        let config_graphics_settings = config_graphics.settings.unwrap_or_else(|| {
            warn!("config.graphics.settings is invalid, using default");
            Default::default()
        });

        // TODO: marker for config change
        let result = Config {
            controls: ConfigControls {
                movement: ConfigControlsMovement {
                    forwards: Input::from_str(config_controls_movement.forwards, "config.controls.movement.forwards", "w"),

                    backwards: Input::from_str(config_controls_movement.backwards, "config.controls.movement.backwards", "s"),

                    left: Input::from_str(config_controls_movement.left, "config.controls.movement.left", "a"),

                    right: Input::from_str(config_controls_movement.right, "config.controls.movement.right", "d"),

                    down: Input::from_str(config_controls_movement.down, "config.controls.movement.down", "l_control"),

                    up: Input::from_str(config_controls_movement.up, "config.controls.movement.up", "space"),

                    speed_up: Input::from_str(config_controls_movement.speed_up, "config.controls.movement.speed_up", "scroll_up"),

                    speed_down: Input::from_str(config_controls_movement.speed_down, "config.controls.movement.speed_down", "scroll_down"),
                },
                sensitivity: ConfigControlsSensitivity {
                    mouse_speed: config_controls_sensitivity.mouse_speed.unwrap_or_else(|| {
                        warn!("config.controls.sensitivity.mouse_speed is invalid, using default");
                        1.0
                    }),

                    movement_speed: config_controls_sensitivity.movement_speed.unwrap_or_else(|| {
                        warn!("config.controls.sensitivity.movement_speed is invalid, using default");
                        1.0
                    }),

                    scroll_speed: config_controls_sensitivity.scroll_speed.unwrap_or_else(|| {
                        warn!("config.controls.sensitivity.scroll_speed is invalid, using default");
                        1.0
                    }),
                },
                engine: ConfigControlsEngine {
                    grab_cursor: Input::from_str(config_controls_engine.grab_cursor, "config.controls.engine.grab_cursor", "g"),

                    wireframe: Input::from_str(config_controls_engine.wireframe, "config.controls.engine.wireframe", "f"),
                },
            },
            graphics: ConfigGraphics {
                settings: ConfigGraphicsSettings {
                    vsync: config_graphics_settings.vsync.unwrap_or_else(|| {
                        warn!("config.graphics.settings.vsync is invalid, using default");
                        true
                    }),
                }
            },
        };

        log::info!("Config initialized");

        result
    }

    pub fn save(&self, config_path: &str) {
        let path = format!("./{}/{}", ::NAME, config_path);

        // TODO: marker for config change
        let reconstructed = option::Config {
            controls: Some(option::ConfigControls {
                movement: Some(option::ConfigControlsMovement {
                    forwards: Some(self.controls.movement.forwards.to_string()),
                    backwards: Some(self.controls.movement.backwards.to_string()),
                    left: Some(self.controls.movement.left.to_string()),
                    right: Some(self.controls.movement.right.to_string()),
                    down: Some(self.controls.movement.down.to_string()),
                    up: Some(self.controls.movement.up.to_string()),
                    speed_up: Some(self.controls.movement.speed_up.to_string()),
                    speed_down: Some(self.controls.movement.speed_down.to_string()),
                }),
                sensitivity: Some(option::ConfigControlsSensitivity {
                    mouse_speed: Some(self.controls.sensitivity.mouse_speed),
                    movement_speed: Some(self.controls.sensitivity.movement_speed),
                    scroll_speed: Some(self.controls.sensitivity.scroll_speed),
                }),
                engine: Some(option::ConfigControlsEngine {
                    grab_cursor: Some(self.controls.engine.grab_cursor.to_string()),
                    wireframe: Some(self.controls.engine.wireframe.to_string()),
                }),
            }),
            graphics: Some(option::ConfigGraphics {
                settings: Some(option::ConfigGraphicsSettings {
                    vsync: Some(self.graphics.settings.vsync),
                }),
            }),
        };
        let serialized = toml::to_string(&reconstructed).unwrap_or_else(|err| {
            warn!("{}", err);
            String::new()
        });

        let mut file = OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap_or_else(|err| ::error_close!("{}", err));

        // delete everything in file
        file.set_len(0)
            .unwrap_or_else(|err| ::error_close!("{}", err));

        file.write_all(serialized.as_bytes())
            .unwrap_or_else(|err| ::error_close!("{}", err));
    }
}
