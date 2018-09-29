use std::{
    fs::{File, OpenOptions},
    io::{self, SeekFrom, Write},
};
use logger::Logger;
use super::keycode::Input;
use ::toml;
use ::winit::{
    VirtualKeyCode, ElementState, MouseButton,
};

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
    }
}


impl Config {
    pub fn new(logger: &mut Logger, confpath: &str) -> Self {
        let config = option::new(logger, confpath);

        // TODO: marker for config change
        let config_controls = config.controls
            .unwrap_or_else(|| {
                logger.warning("ConfigChecker", "config.controls is invalid, using default");
                Default::default()
            });
        let config_controls_movement = config_controls.movement
            .unwrap_or_else(|| {
                logger.warning("ConfigChecker", "config.controls.movement is invalid, using default");
                Default::default()
            });
        let config_controls_sensitivity = config_controls.sensitivity
            .unwrap_or_else(|| {
                logger.warning("ConfigChecker", "config.controls.sensitivity is invalid, using default");
                Default::default()
            });
        let config_controls_engine = config_controls.engine
            .unwrap_or_else(|| {
                logger.warning("ConfigChecker", "config.controls.engine is invalid, using default");
                Default::default()
            });

        // TODO: marker for config change
        let result = Config {
            controls: ConfigControls {
                movement: ConfigControlsMovement {
                    forwards: Input::from_str(config_controls_movement.forwards.unwrap_or_else(|| {
                        logger.warning("ConfigChecker", "config.controls.movement.forwards is invalid, using default");
                        String::from("w")
                    })).warn_none(logger, "config.controls.movement.forwards"),

                    backwards: Input::from_str(config_controls_movement.backwards.unwrap_or_else(|| {
                        logger.warning("ConfigChecker", "config.controls.movement.backwards is invalid, using default");
                        String::from("s")
                    })).warn_none(logger, "config.controls.movement.backwards"),

                    left: Input::from_str(config_controls_movement.left.unwrap_or_else(|| {
                        logger.warning("ConfigChecker", "config.controls.movement.left is invalid, using default");
                        String::from("a")
                    })).warn_none(logger, "config.controls.movement.left"),

                    right: Input::from_str(config_controls_movement.right.unwrap_or_else(|| {
                        logger.warning("ConfigChecker", "config.controls.movement.right is invalid, using default");
                        String::from("d")
                    })).warn_none(logger, "config.controls.movement.right"),

                    down: Input::from_str(config_controls_movement.down.unwrap_or_else(|| {
                        logger.warning("ConfigChecker", "config.controls.movement.down is invalid, using default");
                        String::from("l_control")
                    })).warn_none(logger, "config.controls.movement.down"),

                    up: Input::from_str(config_controls_movement.up.unwrap_or_else(|| {
                        logger.warning("ConfigChecker", "config.controls.movement.up is invalid, using default");
                        String::from("space")
                    })).warn_none(logger, "config.controls.movement.up"),

                    speed_up: Input::from_str(config_controls_movement.speed_up.unwrap_or_else(|| {
                        logger.warning("ConfigChecker", "config.controls.movement.speed_up is invalid, using default");
                        String::from("scroll_up")
                    })).warn_none(logger, "config.controls.movement.speed_up"),

                    speed_down: Input::from_str(config_controls_movement.speed_down.unwrap_or_else(|| {
                        logger.warning("ConfigChecker", "config.controls.movement.speed_down is invalid, using default");
                        String::from("scroll_down")
                    })).warn_none(logger, "config.controls.movement.speed_down"),
                },
                sensitivity: ConfigControlsSensitivity {
                    mouse_speed: config_controls_sensitivity.mouse_speed.unwrap_or_else(|| {
                        logger.warning("ConfigChecker", "config.controls.sensitivity.mouse_speed is invalid, using default");
                        1.0
                    }),

                    movement_speed: config_controls_sensitivity.movement_speed.unwrap_or_else(|| {
                        logger.warning("ConfigChecker", "config.controls.sensitivity.movement_speed is invalid, using default");
                        1.0
                    }),

                    scroll_speed: config_controls_sensitivity.scroll_speed.unwrap_or_else(|| {
                        logger.warning("ConfigChecker", "config.controls.sensitivity.scroll_speed is invalid, using default");
                        1.0
                    }),
                },
                engine: ConfigControlsEngine {
                    grab_cursor: Input::from_str(config_controls_engine.grab_cursor.unwrap_or_else(|| {
                        logger.warning("ConfigChecker", "config.controls.engine.grab_cursor is invalid, using default");
                        String::from("g")
                    })).warn_none(logger, "config.controls.engine.grab_cursor"),

                    wireframe: Input::from_str(config_controls_engine.wireframe.unwrap_or_else(|| {
                        logger.warning("ConfigChecker", "config.controls.engine.wireframe is invalid, using default");
                        String::from("f")
                    })).warn_none(logger, "config.controls.engine.wireframe"),
                },
            },
        };

        logger.info("Config", "Config initialized");

        result
    }

    pub fn save(&self, logger: &mut Logger, config_path: &str) {
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
                    wireframe: Some(self.controls.engine.wireframe.to_string())
                })
            }),
        };
        let serialized = toml::to_string(&reconstructed)
            .unwrap_or_else(|err| {
                logger.warning("ConfigGen", err);
                String::new()
            });
        
        let mut file = OpenOptions::new()
            .write(true).open(&path)
            .unwrap_or_else(|err| logger.error("ConfigSave", err));

        // delete everything in file
        file.set_len(0)
            .unwrap_or_else(|err| logger.error("ConfigSave", err));

        file.write(serialized.as_bytes())
            .unwrap_or_else(|err| logger.error("ConfigSave", err));
    }
}
