// https://github.com/Friz64/ecsvulk/issues/2
macro_rules! gen_config {
    ( $( $main_name:ident $main_name_short:ident { $( $sub_name:ident $sub_name_short:ident { $( $field_name: ident $option_type:ident $real_type:ident $default:expr )* } ),* } ),* ) => (
        trait UpdateConfigHack {
            fn update_key(&mut self, _: Option<VirtualKeyCode>, _: ElementState);
            fn update_mouse(&mut self, _: MouseButton, _: ElementState);
            fn update_scroll(&mut self, _: f32);
            fn update_status(&mut self);
        }
        
        impl UpdateConfigHack for f32 {
            fn update_key(&mut self, _: Option<VirtualKeyCode>, _: ElementState) {}
            fn update_mouse(&mut self, _: MouseButton, _: ElementState) {}
            fn update_scroll(&mut self, _: f32) {}
            fn update_status(&mut self) {}
        }
        
        mod option {
            use super::*;

            #[derive(Debug, Deserialize, Serialize)]
            #[allow(non_snake_case)]
            pub struct Config {
                $(
                    pub $main_name_short: Option<$main_name>,
                )*
            }

            impl Default for Config {
                fn default() -> Self {
                    Self {
                        $(
                            $main_name_short: Some($main_name {
                                $(
                                    $sub_name_short: Some($sub_name {
                                        $(
                                            $field_name: Some($default),
                                        )*
                                    }),
                                )*
                            }),
                        )*
                    }
                }
            }
            
            $(
                #[derive(Debug, Deserialize, Serialize)]
                #[allow(non_snake_case)]
                pub struct $main_name {
                    $(
                        pub $sub_name_short: Option<$sub_name>,
                    )*
                }

                impl Default for $main_name {
                    fn default() -> Self {
                        Self {
                            $(
                                $sub_name_short: Some($sub_name {
                                    $(
                                        $field_name: Some($default),
                                    )*
                                }),
                            )*
                        }
                    }
                }

                $(
                    #[derive(Debug, Deserialize, Serialize)]
                    pub struct $sub_name {
                        $(
                            pub $field_name: Option<$option_type>,
                        )*
                    }

                    impl Default for $sub_name {
                        fn default() -> Self {
                            Self {
                                $(
                                    $field_name: Some($default),
                                )*
                            }
                        }
                    }
                )*
            )* 

            pub fn new(config_path: &str) -> Config {
                use std::io::{Read, Seek};

                let path = format!("./{}/{}", ::NAME, config_path);
                let default = Config::default();

                let mut file = File::open(&path)
                    .unwrap_or_else(|err| match err.kind() {
                        io::ErrorKind::NotFound => {
                            log::warn!("Config file missing, creating new one");

                            // file::create doesn't allow reading
                            // https://doc.rust-lang.org/std/fs/struct.OpenOptions.html
                            let mut file = OpenOptions::new()
                                .read(true).write(true).create(true).open(&path)
                                .unwrap_or_else(|err|
                                    ::error_close!("{}", err)
                                );

                            let serialized = toml::to_string(&default)
                                .unwrap_or_else(|err| {
                                    warn!("{}", err);
                                    String::new()
                                });

                            file.write(serialized.as_bytes())
                                .map_err(|err| {
                                    warn!("{}", err);
                                }).ok();

                            file
                        },
                        _ => ::error_close!("{}", err),
                    });

                // read the file from start because we just wrote to it
                // https://doc.rust-lang.org/std/io/trait.Seek.html#tymethod.seek
                file.seek(SeekFrom::Start(0))
                    .unwrap_or_else(|err| {
                        warn!("{}", err);
                        Default::default()
                    });

                let mut contents = String::new();
                file.read_to_string(&mut contents)
                    .unwrap_or_else(|err| {
                        warn!("{}", err);
                        Default::default()
                    });

                toml::from_str(&contents)
                    .unwrap_or_else(|err| {
                        warn!("{} - USING DEFAULT CONFIG INSTEAD", err);
                        Config::default()
                    })
            }
        }

        #[derive(Debug)]
        #[allow(non_snake_case)]
        pub struct Config {
            $(
                pub $main_name_short: $main_name,
            )*
        }
        
        $(
            #[derive(Debug)]
            #[allow(non_snake_case)]
            pub struct $main_name {
                $(
                    pub $sub_name_short: $sub_name,
                )*
            }

            $(
                #[derive(Debug)]
                pub struct $sub_name {
                    $(
                        pub $field_name: $real_type,
                    )*
                }
            )*
        )*

        pub trait UpdateConfig {
            fn update_keys(&mut self, keycode: Option<VirtualKeyCode>, state: ElementState);
            fn update_mouse(&mut self, button: MouseButton, state: ElementState);
            fn update_scroll(&mut self, delta: f32);
            fn update_status(&mut self);
        }

        impl UpdateConfig for Config {
            fn update_keys(&mut self, keycode: Option<VirtualKeyCode>, state: ElementState) {
                $($($(
                    self.$main_name_short.$sub_name_short.$field_name.update_key(keycode, state);
                )*)*)*
            }

            fn update_mouse(&mut self, button: MouseButton, state: ElementState) {
                $($($(
                    self.$main_name_short.$sub_name_short.$field_name.update_mouse(button, state);
                )*)*)*
            }
            
            fn update_scroll(&mut self, delta: f32) {
                $($($(
                    self.$main_name_short.$sub_name_short.$field_name.update_scroll(delta);
                )*)*)*
            }

            fn update_status(&mut self) {
                $($($(
                    self.$main_name_short.$sub_name_short.$field_name.update_status();
                )*)*)*
            }
        }
    )
}