import React from "react";
import {
    Box,
    Button,
    Slider,
    Radio,
    Group,
    Text,
    Stack,
    Divider,
    Title,
} from "@mantine/core";

const ToolBox = ({ tool, setTool, undo, clearAll }) => {
    return (
        <Box
            sx={{
                width: 250,
                backgroundColor: "#f8f9fa",
                padding: 20,
                borderRight: "1px solid #ddd",
                display: "flex",
                flexDirection: "column",
                gap: 20,
            }}
        >
            {/* Title */}
            <Title order={4} align="center" sx={{ fontWeight: 700 }}>
                Toolbox
            </Title>

            {/* Brush Settings */}

            <Divider my='sm' />

            {/* Fill Type */}
            <Stack spacing="sm">
                <Text weight={500}>Fill Type</Text>
                <Group>
                    <Radio.Group
                        name="fillType"
                        value={tool}
                        onChange={(value) => { setTool(value) }}
                    >
                        <Box mb="sm">
                            <Radio value="stroke" label="Stroke Fill" />
                        </Box>
                        <Box mb="sm">
                            <Radio value="polygon" label="Polygon Fill" />
                        </Box>
                    </Radio.Group>
                </Group>
            </Stack>

            <Divider my='md' />

            {/* Tool Actions */}
            <Stack spacing="xs">
                <Text weight={500}>Actions</Text>
                <Button color="gray" variant="light" onClick={undo}>
                    Undo
                </Button>
                <Button color="red" variant="light" onClick={clearAll}>
                    Clear All
                </Button>
            </Stack>

            <Divider my='md' />

            {/* Tool Tip */}
            <Stack spacing="xs">
                <Text size="xl">Tool Tips</Text>
                <Text size="md">
                    <b>Stroke Fill:</b> Draw shape by clicking and dragging.
                </Text>
                <Text size="md">
                    <b>Polygon Fill:</b> Click multiple points to draw a polygon. <u>Double click</u> to close the polygon.
                </Text>
            </Stack>
        </Box>
    );
};

export default ToolBox;